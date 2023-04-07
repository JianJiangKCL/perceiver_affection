import pandas as pd
import numpy as np
import wandb
import itertools
api = wandb.Api()
# because there are some values are not reported for pre and post and in-process baselines, this file is used to process csv results

# sorted runs by modalities
# runs = sorted(runs, key=lambda x: x.config["modalities"])
def latex_table(fn):
    def wrapper(*args, **kwargs):
        # print content from table_template.tex
        with open("latex/table_template_personality.tex", "r") as f:
        # with open("latex/table_template_v3.tex", "r") as f:
            # print if not \bottomrule
            # iterate over lines
            line = ''
            pre_table = ''
            caption_name = kwargs['caption_name']
            while "bottomrule" not in line:

                if "\\caption{" in line and caption_name is not None:
                    line = caption_name + "\n"
                pre_table += line
                line = f.readline()
            print(pre_table)
            fn(*args, **kwargs)
            print(line)
            # read() the rest of the file
            print(f.read())
            print('\n \n')
            return None
    return wrapper




@latex_table
def print_sorted_runs(df, to_report_metrics, to_separate_metrics, caption_name=None):
    def modifiy_item(item):
        # replace audio with Audio
        item = item.replace("%", "")
        item = item.replace("audio", "Audio")
        # replace facebody with FaceBody
        item = item.replace("facebody", "Video")
        # replace senti,speech,time with Textual
        item = item.replace("senti,speech,time", "Textual")
        # replace senti,speech,text,time with  BERT, Textual
        item = item.replace("senti,speech,text,time", "BERT, Textual")
        # # replace text with BERT, except \textcolor
        item = item.replace("text", "BERT")
        item = item.replace("\BERT", "\\text")
        return item
    items = []
    modalities = df['modalities']
    for modality in modalities:
        item = ''
        modality_set = modality.split('_')
        modality_set = sorted(modality_set)
        modality_set = ','.join(modality_set)
        item += modality_set + ' & '
        for metric in to_report_metrics:
            value = df[df['modalities'] == modality][metric].values[0]
            if metric in to_separate_metrics:
                item += str(round(value, 2)) + ' &'
            else:
                value = round(value * 100, 2)
                if value < 80:
                    item += '\\textcolor{red}{' + str(value) + '} & '
                else:
                    item += str(value) + ' & '
        # remove the last &
        item = item[:-2]
        item += '\\\\'
        items.append(item)
        print(item)
k=1


to_separate_metrics = ["val_mse", "val_mse_O", "val_mse_C", "val_mse_E", "val_mse_A", "val_mse_N", "val_MSE_O", "val_MSE_C", "val_MSE_E", "val_MSE_A", "val_MSE_N"]


to_report_metrics = ["val_mse", "val_mse_O", "gender_val_DIR_0","age_val_DIR_0", "val_mse_C", "gender_val_DIR_1", "age_val_DIR_1", "val_mse_E", "gender_val_DIR_2", "age_val_DIR_2", "val_mse_A", "gender_val_DIR_3", "age_val_DIR_3", "val_mse_N", "gender_val_DIR_4",  "age_val_DIR_4"]

key_name_test = 'test'
to_report_metrics = [x.replace('val', key_name_test) for x in to_report_metrics]
to_separate_metrics = [x.replace('val', key_name_test) for x in to_separate_metrics]


csv_path = "H:/project/perceiver_affection/results/single_stage_fair_pre.csv"

df = pd.read_csv(csv_path)
caption_ = "single stage fair pre"
caption_name = "\\caption{" + caption_ + "}"
print_sorted_runs(df, to_report_metrics, to_separate_metrics, caption_name=caption_name)

