import matplotlib
import numpy as np
import argparse
import collections
import matplotlib.pyplot as plt

def main_cls(cls):
    coarse=cls[0]
    assert coarse.isnumeric(), cls
    if len(cls)>1 and cls[1] in ("<",">"):
        coarse+="<>"
    return coarse

def read_results(inp):
    counter={}
    for line in inp:
        line=line.strip()
        cls,position,sent_L,sent_R=line.split("\t")
        if cls.startswith("4>") and "-rev" in cls:
            continue
        if cls.startswith("4<") and "-rev" not in cls:
            continue
        if position=="None":
            position=None
        else:
            position=int(position)
        coarse_cls=main_cls(cls)
        if position is None:
            pos="NA"
        else:
            for cutoff in (1,10,100,1000,2048):
                if position-1<cutoff: #0 is always the qsentence itself so -1 to adjust
                    pos=str(cutoff)
                    break
            else:
                assert False    
        counter.setdefault(coarse_cls,[]).append(pos)
    for k,pos_list in counter.items():
        counter[k]=collections.Counter(pos_list)
    result={}
    for k,pos_list in counter.items():
        tot=sum(v for k,v in pos_list.items())
        res_list=[]
        for r in ("1","10","100","1000","2048","NA"):
            res_list.append(pos_list[r]/tot*100)
        result[k]=res_list
    return result

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-results",help='Result file for BERT')
    parser.add_argument("--sbert-results",help='Result file for SBERT') 

    args = parser.parse_args()
    counter_bert=read_results(open(args.bert_results))
    counter_sbert=read_results(open(args.sbert_results))

    results={}
    for k,v in counter_bert.items():
        results["bert-"+k]=v
    for k,v in counter_sbert.items():
        results["sbert-"+k]=v
    
    category_names = ['Top1', 'Top10', 'Top100', 'Top1000', 'Top2048', 'NA']

    def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = ["sbert-4","sbert-4<>","sbert-3","bert-4","bert-4<>","bert-3"]
        #labels = ["sbert-4","bert-4","sbert-4<>","bert-4<>","sbert-3","bert-3"]
        data = np.array(list(results[k] for k in labels))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.85, 0.15, data.shape[1]))

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize='small')

        return fig, ax


    survey(results, category_names)
    plt.show()


