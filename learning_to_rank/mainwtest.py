import lightgbm as lgb

# Note that we have convert the original raw data into a pure libsvm format.
# For more details, pls refer to: https://github.com/guolinke/boosting_tree_benchmarks/tree/master/data
infile_train = "files.train"
infile_test = "files.test"

train_data = lgb.Dataset(infile_train)
test_data = lgb.Dataset(infile_test)

# Set group info.
# We can igonre the step if *.query files exist with input files in the same dir.
train_group_size = [l.strip("\n") for l in open(infile_train + ".query")]
valid_group_size = [l.strip("\n") for l in open(infile_test + ".query")]
train_data.set_group(train_group_size)
test_data.set_group(valid_group_size)

# Parameters are borrowed from the official experiment doc:
# https://lightgbm.readthedocs.io/en/latest/Experiments.html
param = {
    "task": "train",
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 100,
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": .1,
    "num_threads": 16
}

res = {}
bst = lgb.train(
    param, train_data,
    valid_sets=[train_data, test_data], valid_names=["train", "test"],
    num_boost_round=1000, evals_result=res, verbose_eval=50)

bst.save_model("model_large.txt")
