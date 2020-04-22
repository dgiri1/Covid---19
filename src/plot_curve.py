from matplotlib import pyplot as plt
from shared_utils import *
from plot_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict

diseases = ["covid"]

with open(f'{root_dir}/data/states/states_1.pkl',"rb") as f:
    states = pkl.load(f)

xlim = (-110,-70)
ylim = (25,40)

stateByName = OrderedDict([('North Carolina', '0400000US37'),('Texas', '0400000US48'),
                            ('Florida', '0400000US12'), ('Georgia', '0400000US13')])

plot_state_names = {"covid": ['North Carolina','Texas','Florida','Georgia']}

# colors for curves
C1 = "#D55E00"
C2 = "#E69F00"
C3 = "#0073CF"

fig = plt.figure(figsize=(12, 10))
grid = plt.GridSpec(2, 2*len(diseases), top=0.9, bottom=0.2, left=0.07, right=0.97, hspace=0.25, wspace=0.15, height_ratios=[1,1])

for i,disease in enumerate(diseases):
    # Load data
    use_age = True
    use_eastwest = True
    if disease=="covid":
        prediction_region = "South"
        use_eastwest = True

        
    data = load_data(disease, prediction_region, states)
    data = data[data.index < pd.Timestamp(*(2020,4,16))]
    if disease == "covid":
        data = data[data.index >= pd.Timestamp(*(2020,1,22))]
    _, _, _, target = split_data(data)
    state_ids = target.columns

    # Load our prediction samples
    res = load_pred(disease, use_age, use_eastwest)

    prediction_samples = np.reshape(res['y'],(res["y"].shape[0],15,-1))
    prediction_quantiles = np.quantile(prediction_samples,q=[.05,.25,.75,.95],axis=0)

    prediction_mean = pd.DataFrame(data=np.mean(prediction_samples,axis=0), index=target.index, columns=target.columns).sort_index().iloc[:10]
    prediction_q5 = pd.DataFrame(data=prediction_quantiles[0,:,:], index=target.index, columns=target.columns).sort_index().iloc[:10]
    prediction_q25 = pd.DataFrame(data=prediction_quantiles[1,:,:], index=target.index, columns=target.columns).sort_index().iloc[:10]
    prediction_q75 = pd.DataFrame(data=prediction_quantiles[2,:,:], index=target.index, columns=target.columns).sort_index().iloc[:10]
    prediction_q95 = pd.DataFrame(data=prediction_quantiles[3,:,:], index=target.index, columns=target.columns).sort_index().iloc[:10]



    for j,name in enumerate(plot_state_names[disease]):

        ax = fig.add_subplot(grid[j//2,j%2])

        state_id = stateByName[name]
        dates = prediction_mean.index[:10]
        # plot our predictions w/ quartiles

        p_pred=ax.plot_date(dates, prediction_mean[state_id], "-", color=C1, linewidth=2.0, zorder=4)
        p_quant=ax.fill_between(dates, prediction_q25[state_id], prediction_q75[state_id], facecolor=C2, alpha=0.5, zorder=1)
        ax.plot_date(dates,prediction_q25[state_id], ":", color=C2, linewidth=2.0, zorder=3)
        ax.plot_date(dates,prediction_q75[state_id], ":", color=C2, linewidth=2.0, zorder=3)

       
        # plot ground truth
        p_real=ax.plot_date(dates, target.loc[dates][state_id], "k.")
        # ax.axvline(dates[9], lw=2)

        ax.set_title("$"+ name +"$", fontsize=22)
        # if j == 1:
        #     ax.set_xlabel("Time [calendar weeks]", fontsize=22)
        ax.tick_params(axis="both", direction='out', size=6, labelsize=16, length=6)
        plt.setp(ax.get_xticklabels(), visible=j>1, rotation=45)


        ax.autoscale(False)
        p_quant2=ax.fill_between(dates, prediction_q5[state_id], prediction_q95[state_id], facecolor=C2, alpha=0.25, zorder=0)
        ax.plot_date(dates,prediction_q5[state_id], ":", color=C2, alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot_date(dates,prediction_q95[state_id], ":", color=C2, alpha=0.5, linewidth=2.0, zorder=1)

    plt.legend([p_real[0], p_pred[0], p_quant, p_quant2],
    ["reported", "predicted", "$25\%-75\%$ quantile", "$5\%-95\%$ quantile"],
    fontsize=16, ncol=5, loc="upper center", bbox_to_anchor = (0,-0.01,1,1),
        bbox_transform = plt.gcf().transFigure )
    fig.text(0.5, 0.02, "Time [calendar days]", ha='center', fontsize=22)
    fig.text(0.01, 0.46, "Reported/predicted infections", va='center', rotation='vertical', fontsize=22)
    # plt.savefig("../figures/curves_{}_appendix.pdf".format(disease))