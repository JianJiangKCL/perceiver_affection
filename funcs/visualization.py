import pandas as pd
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
# runs = api.runs("jianjiang/perceiver_affection_hyper_trainval")
runs = api.runs("jianjiang/perceiver_affection_spd_trainval_bul")


summary_list, config_list, name_list = [], [], []
to_report_metrics = [ "val_mse", "gender_val_DIR_O", "gender_val_DIR_C", "gender_val_DIR_E", "gender_val_DIR_A", "gender_val_DIR_N", "age_val_DIR_O", "age_val_DIR_C", "age_val_DIR_E", "age_val_DIR_A", "age_val_DIR_N"]
to_separate_metrics = ["val_mse"]
# filter_configs = {"depth":5, "lr": 0.004, "num_latents": 256, "epochs":100}
filter_configs = { "lr": 0.004, "gamma": 10, "epochs":10}
to_report_configs = ["modalities"]

# sorted runs by modalities
# runs = sorted(runs, key=lambda x: x.config["modalities"])
items = []
modalities_names = []
for run in runs:
    flag_filtered = False
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    item = ''
    # filter out runs that do not match the filter metrics
    for key in filter_configs:
        if key in run.config:
            if run.config[key] != filter_configs[key]:
                # print("key", key, "value", run.config[key], "not match", filter_configs[key])
                flag_filtered = True
                break

    if not flag_filtered:
        for key in to_report_configs:
            if key in run.config:
                # process the modality name
                if key == "modalities":
                    modalities = run.config[key]
                    # if is a list, then get the first element

                    if isinstance(modalities, list):
                        modalities = modalities[0]
                    # if contains _ , then split and sort
                    if '_' in modalities:
                        modalities = modalities.split('_')
                        # sort the modalities based alphabetical order
                        modalities = sorted(modalities)
                        # combine them with ','
                        # modalities = ','.join(modalities)
                    # item += modalities + ' & '
                    modalities_names.append(modalities)
                else:
                    item += str(run.config[key]) + ' & '

        for key in to_report_metrics:
            if key in run.summary._json_dict:

                # if key in to_separate_metrics:
                if "val_mse" in key:
                    # keep 4 decimal places
                    item += str(round(run.summary._json_dict[key], 4)) + ' & '
                # times 100, and
                else:
                    value = round(run.summary._json_dict[key] * 100, 2)
                    if value < 80:
                        #  \textcolor{red}{0.7907}
                        item += '\\textcolor{red}{' + str(value) + '} & '
                    else:
                        item += str(value) + ' & '
        # remove the last &
        item = item[:-2]
        # plus \\
        item += '\\\\'
        items.append(item)
        # # replace audio with Audio
        # item = item.replace("audio", "Audio")
        # # replace facebody with FaceBody
        # item = item.replace("facebody", "FaceBody")
        # # replace senti,speech,time with Textual
        # item = item.replace("senti,speech,time", "Textual")
        # # replace senti,speech,text,time with  BERT, Textual
        # item = item.replace("senti,speech,text,time", "BERT, Textual")
        # # # replace text with BERT, except \textcolor
        # item = item.replace("text", "BERT")
        # item = item.replace("\BERT", "\\text")

        # print(item)

# print(modalities_names)
# make each element a list, if it a string add it to a list
modalities_names = [x if isinstance(x, list) else [x] for x in modalities_names]
# print(modalities_names)
# sort with number of elements, in default sorted with alphabetical order
lengths = [len(x) for x in modalities_names]
sorted_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])
sorted_modalities_names = [modalities_names[i] for i in sorted_indices]
sorted_items = [items[i] for i in sorted_indices]
for name, tmp in zip(sorted_modalities_names, sorted_items):
    item = ''
    # covert name to string
    modalities = ','.join(name)
    item += modalities + ' & '
    item += tmp
    # replace audio with Audio
    item = item.replace("audio", "Audio")
    # replace facebody with FaceBody
    item = item.replace("facebody", "FaceBody")
    # replace senti,speech,time with Textual
    item = item.replace("senti,speech,time", "Textual")
    # replace senti,speech,text,time with  BERT, Textual
    item = item.replace("senti,speech,text,time", "BERT, Textual")
    # # replace text with BERT, except \textcolor
    item = item.replace("text", "BERT")
    item = item.replace("\BERT", "\\text")
    print(item)




# runs_df.to_csv("project.csv")