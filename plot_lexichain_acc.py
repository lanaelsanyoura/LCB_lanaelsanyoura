import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#filename = "csv_files/epy_alpha_accuracies_n_epy_disp_wn.csv"
filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_wn_bert_cbow.csv"
#filename = "csv_files/epy_alpha_accuracies_v_bert_lexi_wn.csv"
filename = "csv_files/epy_alpha_accuracies_v_bert_lexi_wn_bert_cbow.csv"
filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_childes_bert_cbow_sent_embedding.csv"
filename = "csv_files/epy_alpha_accuracies_n_bert_lexi_wn_bert_cbow_MOT.csv"

prior = "WordNet" if "wn" in filename else "Childes"

data = pd.read_csv(filename)

fig, ax = plt.subplots()
colors = ['red', 'green', 'blue', 'orange','pink', 'black']
for epy in [-1, 1,2,3]: #[0,1,2,3,4]:
    d_epy = data.loc[data.epy == epy]
    color = colors[epy]
    ax.scatter(x = d_epy["alpha"],
                y = d_epy["Lexical Chaining Acc"]*100,
                edgecolors='w', c=color, label=epy)
    if "sent_embedding" in filename:
         ax.scatter(x = d_epy["alpha"],
               y = d_epy["Sent Bert Bayesian Acc"]*100,alpha=1,marker='>',
                c=color) # CBOW Ber
         ax.scatter(x = d_epy["alpha"],
               y = d_epy["CBOW Bert Bayesian Acc"]*100,alpha=1,marker='^',
                c=color) # CBOW Ber
    elif "bert" in filename:
        ax.scatter(x = d_epy["alpha"],
               y = d_epy["Bert Bayesian Acc"]*100,alpha=1,marker='>',
                c=color) # CBOW Bert
    #ax.scatter(x = d_epy["alpha"],
    #            y = d_epy["% Correct Poly"]*100,alpha=1,marker='>',
    #             c=color) # label=str(epy) + " correct poly"
    #ax.scatter(x = d_epy["alpha"],
    #            y =d_epy["% Correct MFS"]*100, alpha=1, marker="x",
    #            edgecolors='w', c=color) # label=str(epy) + " correct mfs"
    ax.axhline(y=d_epy["MFS Acc"].tolist()[0]*100, color=color, linestyle='--')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=5, fancybox=True, shadow=True)
ax.grid(True)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('{} Prior: Accuracy as Alpha and Entropy Change'.format(prior), y=1.05)
plt.tight_layout()
plt.show()