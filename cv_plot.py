"""
Computer Vision plots
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist_df(df, x_col, x_labels):
    """ To plot histogram from pandas.Series """
    
    print("Plotting histogram for %s..." %x_col)
    print("Total counts:", len(df))
    uniq_labels = sorted(df.unique())
    x_labels = x_labels
    plt.figure(figsize=(20, 20))
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 15})
    #x_pos = np.arange(len(df))  #the label locations
    ax = sns.countplot(df)
    ax.set_xticklabels(x_labels, rotation=40, ha='right')
    plt.tight_layout()
    i = 0
    for l in uniq_labels:
        labels_ttl = (df == l).sum()
        #if labels_ttl < 5:
        #    print(l, labels_ttl)
        plt.text(i, labels_ttl, str(labels_ttl), ha='center')
        i += 1
    plt.show()
    
def plot_bar_prob(list_x, list_y, g_title, x_label, ax):
    """ To plot bar chart from a list """
    
    print("Plotting bar chart for prediction probabilities...")
    
    s_ax = ax
    x_labels = list_x
    #x_pos = np.arange(len(list_x))  #the label locations
    sns.barplot(x=list_x, y=list_y, ax=s_ax)
    s_ax.set_title(g_title)
    s_ax.set_xlabel(x_label)
    s_ax.set_ylabel("probabilities")
    s_ax.set_xticklabels(x_labels, rotation=40, ha='right')
    plt.tight_layout()
    i = 0
    for i in range(len(list_y)):
        s_ax.text(i, list_y[i], str(list_y[i]), ha='center')
        i += 1

def plot_donut(the_num, the_title, sh, lo):
    #create data
    the_data=[the_num, (1-the_num)]
    
    #add the subplot and create pie
    plt.subplot2grid(sh, lo)
    plt.title(the_title)
    wedges, texts = plt.pie(the_data, wedgeprops=dict(width=0.5), colors=['b', 'w'], startangle=-270, counterclock=False)
    the_text = '{0:.2f}%'.format(the_data[0] * 100)
    plt.text(-0.2, 0, the_text, fontsize=15)

def plot_prec_rec(dict_cr, tag_id, list_class):
    """ To plot Precision-Recall curve """
    
    plt.figure(figsize=(20, 8))
    plt.rcParams.update({'font.size': 13})
    list_prec = []
    list_rec = []

    for tid in tag_id:
        scores = dict_cr.get(str(tid))
        if scores == None:
            prec = 0.0
            rec = 0.0
        else:
            prec = round(scores['precision'], 2)
            rec = round(scores['recall'], 2)     
        #print("tid:", tid, prec, rec)
        list_prec.append(prec)
        list_rec.append(rec)

    if len(list_class) != 0:
        x_axis = [list_class[i] for i in tag_id]
    else:
        x_axis = tag_id
    plt.plot(x_axis, list_prec, '-gD', markersize=12, label='Precision')
    plt.plot(x_axis, list_rec, '--ro', label='Recall')
    plt.xticks(rotation=90)
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('Bird Class')
    plt.ylabel('Scores')
    plt.title('Precision Recall scores per class')
    plt.legend()
    plt.show()
