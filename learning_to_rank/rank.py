import lightgbm as lgb

bst = lgb.Booster(model_file='model.txt')
#mdl = lgb.LGBMModel(model_file='model.txt')
#y_pred = bst.predict("files.val", num_iteration=bst.best_iteration)

validation_data = lgb.Dataset("files.val")
#bst.eval(validation_data, "validation")

bst.add_valid(validation_data, "validation")

#lgb.plot_metric(mdl, metric="NDCG@10")

#cur_query = None
#original_file = open("msmarco-doctrain-top100", "r")
#i = 0
#top100list = []

# output_file = open("ranking.out", "w")
#
# for i in range(len(y_pred)):
#     output_file.write(str(y_pred[i]) + "\n")
    # lst = line.split(' ')
    # if lst[0] != cur_query:
    #     cur_query = lst[0]
    # i = i+1

#output_file.close()
