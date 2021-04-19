import matplotlib.pyplot as plt
from mgwr.utils import shift_colormap,truncate_colormap
import numpy
import statsmodels.api as sm


def gwr_param_plots(result, gdf, names=[], filter_t=False, subplots=(2,2), figsize=(10,20)):
    #Size of the plot. Here we have a 2 by 2 layout.
    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    axs = axs.ravel()
    k = gwr_results.k
    #The max and min values of the color bar.
    vmin = -0.8
    vmax = 0.8
    cmap = cm.get_cmap("coolwarm")
    norm = colors.BoundaryNorm(np.arange(-0.8,0.9,0.1),ncolors=256)
    if (vmin < 0) & (vmax < 0):
        cmap = truncate_colormap(cmap, 0.0, 0.5)
    elif (vmin > 0) & (vmax > 0):
        cmap = truncate_colormap(cmap, 0.5, 1.0)
    else:
        cmap = shift_colormap(cmap, start=0.0, midpoint=1 - vmax/(vmax + abs(vmin)), stop=1.)    
    for j in range(k): 
        pd.concat([gdf,pd.DataFrame(np.hstack([result.params,result.bse]))],axis=1).plot(ax=axs[j],column=j,vmin=vmin,vmax=vmax,
                   cmap="bwr",norm=norm,linewidth=0.1,edgecolor='white')
        axs[j].set_title("Parameter estimates of \n" + names[j],fontsize=10)
        if filter_t:
            rslt_filtered_t = result.filter_tvals()
            if (rslt_filtered_t[:,j] == 0).any():
                gdf[rslt_filtered_t[:,j] == 0].plot(color='lightgrey', ax=axs[j],linewidth=0.1,edgecolor='white')
        #plt.axis('off')
    fig = axs[j].get_figure()
    cax = fig.add_axes([0.99, 0.2, 0.025, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    fig.colorbar(sm, cax=cax)

def q_q_plot(res):
    fig, axes = plt.subplots(1, 2,figsize=(12,4))
    axes[0].hist(res)   # code from class notebook wouldn't work saying ndarray had no hist attribute
    sm.qqplot(res,line='q',ax=axes[1])