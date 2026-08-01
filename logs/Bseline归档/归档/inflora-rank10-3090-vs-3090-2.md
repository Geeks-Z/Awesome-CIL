# inflora-rank10-3090 vs inflora-rank10-3090-2

## 结论

`历史日志/inflora-rank10-3090.out` **没有完全覆盖** `历史日志/inflora-rank10-3090-2.out` 的结果。

- 从实验设置覆盖看：`inflora-rank10-3090.out` 包含 `35` 个 setting，`inflora-rank10-3090-2.out` 包含 `28` 个 setting，且 `inflora-rank10-3090-2.out` **没有**任何 `inflora-rank10-3090.out` 不包含的 setting。
- 但在两者共有 setting 中，存在 `1` 个 setting 的结果不一致，因此不能认为前者对后者“结果完全覆盖”。

## Setting 覆盖差异

| 类型 | method | dataset | init_cls | increment |
| --- | --- | --- | ---: | ---: |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | omnibenchmark | 20 | 20 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | omnibenchmark | 30 | 30 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | omnibenchmark | 150 | 5 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | omnibenchmark | 150 | 10 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | omnibenchmark | 150 | 30 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | vtab | 5 | 5 |
| 仅 `inflora-rank10-3090.out` 包含 | inflora | vtab | 10 | 10 |

`inflora-rank10-3090-2.out` 独有 setting：无。

## 共有 Setting 的结果差异

| method | dataset | init_cls | increment | 3090 A_L | 3090-2 A_L | Last 差值 | 3090 A_bar | 3090-2 A_bar | Average 差值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inflora | omnibenchmark | 10 | 10 | 60.74 | 86.67 | -25.93 | 72.60 | 92.97 | -20.37 |

## 统计摘要

| 文件 | unique setting 数 |
| --- | ---: |
| `历史日志/inflora-rank10-3090.out` | 35 |
| `历史日志/inflora-rank10-3090-2.out` | 28 |
