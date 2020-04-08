import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# filename = "csv_files/epy_alpha_accuracies_n_epy_disp_wn.csv"
#filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_childes1.csv"
filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_wn_bert_cbow_CHI.csv"
#filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_wn_bert_cbow_sent_embedding.csv"
prior = "WordNet" if "wn" in filename else "Childes"
data = pd.read_csv(filename)

fig, ax = plt.subplots()
lex_accuracy_list = []
cbow_bert_accuracy_list = []
sent_bert_accuracy_list = []
mfs_acc = 0
alpha_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9,1]
print("alpha,lex_acc, cbow_bert_acc,sent_bert_acc, mfs_acc" )
for alpha in alpha_range:
    d_epy = data.loc[data.alpha == alpha]

    lex_acc = np.mean(d_epy["Lexical Chaining Acc"])*100
    if "sent_embedding" in filename:
        cbow_bert_acc = np.mean(d_epy["CBOW Bert Bayesian Acc"])*100
        cbow_bert_accuracy_list.append(cbow_bert_acc)

        sent_bert_acc = np.mean(d_epy["Sent Bert Bayesian Acc"])*100
        sent_bert_accuracy_list.append(sent_bert_acc)
    elif "bert" in filename:
        cbow_bert_acc = np.mean(d_epy["Bert Bayesian Acc"])*100
        cbow_bert_accuracy_list.append(cbow_bert_acc)
    lex_accuracy_list.append(lex_acc)
    mfs_acc = np.mean(d_epy["MFS Acc"])*100
    print("acc", alpha,lex_acc, cbow_bert_acc, mfs_acc ) # sent_bert_acc
    print("Lexi", alpha, np.mean(d_epy["% Correct Poly"]), np.mean(d_epy["% Correct MFS"]))
    print("Bert", alpha, np.mean(d_epy["Bert % Correct Poly"]), np.mean(d_epy["Bert % Correct MFS"]))
ax.scatter(x = alpha_range,
            y = lex_accuracy_list,
           label="Lexical Chaining Bayesian" )
if "sent_embedding" in filename:
    ax.scatter(x = alpha_range,
               y = cbow_bert_accuracy_list,
                marker=">", color="green", label="CBOW BERT Lexical Bayesian")
    ax.scatter(x = alpha_range,
               y = sent_bert_accuracy_list,
                marker=">", color="red", label="Sent BERT Lexical Bayesian")

elif "bert" in filename:
    ax.scatter(x = alpha_range,
               y = cbow_bert_accuracy_list,
                marker=">", color="red", label="Sent BERT Lexical Bayesian")

ax.axhline(y=mfs_acc, linestyle='--', label="Most Frequent Sense")
ax.legend(ncol=1, fancybox=True, shadow=True)
ax.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Child Speech for Nouns ({} Prior)'.format(prior), y=1.05) # WSD Average Accuracy as Alpha Changes
plt.tight_layout()
plt.show()