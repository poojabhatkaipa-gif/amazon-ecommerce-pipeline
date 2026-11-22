# Model performance (leave-last-out)

Eval users: 10000 | train pairs: 725926 | catalogue: 89973 items | unreachable test items: 0.040%

Random-ranking recall@10 would be about 0.00011.

| run           |   recall@5 |   recall@10 |   recall@20 |   precision@10 |   ndcg@10 |   ndcg@20 |   item_coverage@10 | run_id                           |
|:--------------|-----------:|------------:|------------:|---------------:|----------:|----------:|-------------------:|:---------------------------------|
| popularity    |     0.0002 |      0.0003 |      0.0004 |          3e-05 |   0.00011 |   0.00014 |            0.0002  | bf5fecd4dd5d408cbddaeeb0e099d94c |
| svd_k16       |     0.0001 |      0.0002 |      0.0003 |          2e-05 |   7e-05   |   0.0001  |            0.005   | 555a66b785b94523af4bafecefb4cfa7 |
| svd_k32       |     0      |      0.0001 |      0.0003 |          1e-05 |   4e-05   |   9e-05   |            0.00916 | 5834596b9aa64863a9b9edb66c1dd102 |
| svd_k64       |     0      |      0      |      0.0001 |          0     |   0       |   2e-05   |            0.01722 | b8909a29741640f5b19618a271d162f1 |
| content_based |     0      |      0.0001 |      0.0002 |          1e-05 |   3e-05   |   5e-05   |            0.24114 | e497b98913c349808900554ae23d9d37 |
