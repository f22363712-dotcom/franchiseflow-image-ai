# FranchiseFlow 状态存储迁移契约

这份文档和 `prototype/franchiseflow/storage.py` 一起定义存储层的下一步边界。当前服务仍使用每个任务一个 JSON 文件、批次一个 JSON 文件，以及一个幂等索引。`JsonStorageRepository` 只是把这三组已有能力放在同一个接口下，当前应用尚未切换到它。

## 当前代码状态

`storage_runtime.py` 提供 `build_repository(...)` runtime factory。调用方传入
`backend="json"` 或 `backend="sqlite"`，工厂分别返回
`JsonStorageRepository` 或 `SQLiteStorageRepository`；其他值会直接抛出
`ValueError`。SQLite 未传 `sqlite_path` 时使用
`<output_dir>/franchiseflow.db`，显式传入路径则使用调用方指定的数据库文件。
这个切片只统一了实例构造，不改变现有服务的启动路径。

`storage.py` 已提供 `SQLiteStorageRepository`，使用标准库 `sqlite3` 建立
`jobs`、`batches` 和 `idempotency_keys` 表。它启用 WAL、外键检查和忙等待，
把完整业务记录保存在 `payload_json`，并以 `tenant_id='local'` 兼容单机部署。
幂等 reservation、commit、release 在事务内执行，唯一键由 SQLite 负责跨实例互斥；
任务记录还带有 `version` 和 `compare_and_swap_job`，可供未来 worker claim 使用。

这段代码目前只通过合同测试和临时数据库测试，`app.py` 仍默认使用 JSON 文件存储，
也没有调用 `storage_runtime.build_repository`。新增工厂不代表应用已经切换后端，
更不代表队列已经迁移到 SQLite。
不要通过设置一个环境变量就假设服务已经自动切换到 SQLite，也不要把当前实现描述成
跨副本队列。真正切换前仍需完成迁移、恢复、租约和回滚演练。

## 当前实现的保证

`JsonStorageRepository` 保留 `job_store.py` 的状态迁移校验、`batch_store.py` 的原子替换和 `idempotency_store.py` 的 pending/committed 两阶段记录。写入先落临时文件并替换目标文件，适合单机进程重启后的恢复。

它明确报告以下限制：

- 文件写入是本机持久化，不能协调多个进程或多个副本抢同一个任务。
- 幂等键没有租户范围，键空间仍是 `(scope, key)`。
- 任务状态和视频文件分开保存，没有事务可以同时提交两者。
- `app.py` 仍需在多步幂等流程外层持有进程锁；接口本身不提供跨进程 claim。

这些限制应当被运行环境继续视为单机部署条件，不能把这个适配器当作生产队列。

## SQLite 目标模型

第一版 SQLite 只存任务控制面，上传视频、分店素材、输出 MP4 继续放在文件或对象存储中。记录中的 `input_key`、`asset_key`、`output_key` 指向这些文件。

```sql
CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  batch_id TEXT,
  status TEXT NOT NULL,
  phase TEXT,
  payload_json TEXT NOT NULL,
  claim_token TEXT,
  lease_until TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX jobs_queue_idx
  ON jobs (status, created_at, job_id);

CREATE TABLE batches (
  batch_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE idempotency_keys (
  tenant_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  key TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  state TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (tenant_id, scope, key)
);
```

现有的 `batch_id`、`brand_profile_id`、场景计划和审计字段保留在 `payload_json`，待查询需求稳定后再拆成列。`tenant_id` 从第一天写入，单机兼容层可使用固定的 `local` 租户。

## 任务 claim、租约和 CAS

任务领取必须发生在一个 SQLite 写事务内。worker 先开启 `BEGIN IMMEDIATE`，用子查询挑选最早的 `queued` 任务，再用条件更新完成 claim：

```sql
UPDATE jobs
SET status = 'running',
    version = version + 1,
    claim_token = :token,
    lease_until = :lease_until,
    updated_at = :now
WHERE job_id = (
  SELECT job_id FROM jobs
  WHERE tenant_id = :tenant_id AND status = 'queued'
  ORDER BY created_at, job_id
  LIMIT 1
)
AND status = 'queued';
```

更新影响一行才算领取成功。完成、失败和人工复核写回时必须带上 `job_id`、`claim_token` 和读取时的 `version`，条件不满足就返回冲突，不能覆盖新 worker 的状态。租约过期的 `running` 任务由恢复循环改回 `queued`，并增加重试计数；超过上限转为 `failed` 并保留错误码。

幂等创建仍采用先 `INSERT ... state='pending'`，再落资源记录，最后把同一行改为 `committed`。主键冲突时读取原 fingerprint，指纹不同返回冲突，相同指纹返回原资源。SQLite 的唯一约束承担跨进程互斥，应用层不再依赖 Python 线程锁。

## 迁移、切换和回滚

1. **影子导入**。停止写入或在维护窗口内扫描 `outputs/*/job.json`、`outputs/batches/*.json` 和 `idempotency.json`，按原始时间戳导入 SQLite。导入脚本遇到无效 JSON 时中止并列出文件，不跳过损坏记录。
2. **契约对照**。用 `StorageRepository` 的同一组读取测试分别访问 JSON 和 SQLite，比较任务状态、批次子任务顺序、幂等状态和关键审计字段。
3. **单写切换**。增加 `FRANCHISEFLOW_STORAGE=json|sqlite` 开关。先只让 SQLite 提供读取，确认一轮重启恢复后，再把新写入切到 SQLite；切换时保留一个写屏障时间戳。
4. **观察窗口**。检查队列重复领取、租约回收、幂等重试、批次下载和人工复核。文件输出仍由现有目录校验保护。
5. **回滚**。只允许回滚到写屏障之前的 JSON 快照。切回 JSON 后拒绝写入切换期间新建的任务，或先生成带新任务的 JSON 快照再恢复服务。不要对 JSON 和 SQLite 做双向实时同步，否则无法判断哪个状态获胜。

第一版不做在线 JSON/SQLite 双写，也不在没有租户鉴权的情况下开放多副本。先把 `app.py` 的 `load_job`、`save_job`、`load_batch`、`save_batch` 和幂等调用逐段替换为仓库接口，再引入真实 SQLite 实现；每次替换都保留现有测试并增加对应的 SQLite 合同测试。

