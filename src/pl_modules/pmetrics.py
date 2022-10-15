
import torch

def unnorm(x,maxx,minx):
    y = x.clone()
    for j in range(y.shape[0]):
        y[j] = x[j]*(maxx[j]-minx[j])+minx[j]
    return y

def profit_meas(test_stock_data, look_back, stock_dict, budg0=1000, invest_time=365, tax=0.001, strategy="Long-only", predictor_mode="coinflip", predictor=None, do_print=False, delay=0, training=False, st=True, inc_based=True, normalize=False, maxc=None, minc=None, use_nopen=False, mino=None, maxo=None, split_date=None):

    if stock_dict is None:
        stock_dict = [f"???-{i}" for i in range(test_stock_data.shape[0])]
    test_stock_data = test_stock_data[:,delay:,:]
    strategy = strategy #"Long-only"  #"Long-short"
    profit = {}
    budg = torch.zeros(test_stock_data.shape[0])+budg0
    tax = tax
    if invest_time is None:
        invest_time = test_stock_data.shape[1]
    payed = {}
    plot_predict = {}
    avg_prof_metrics = []
    nor_avg_prof_met = []
    avg_growth = []

    for i in range(test_stock_data.shape[0]):
        plot_predict[i] = []
        profit[i] = []
        payed[i] = torch.tensor(0.)
        avg_prof_metrics.append([])
        nor_avg_prof_met.append([])
        avg_growth.append([])

    for i in range(min(test_stock_data.shape[1]-look_back-1,invest_time+1)):
        if inc_based and i==0: continue

        x_in = test_stock_data[:,i:i+look_back,:]



        if normalize:
            next_open = test_stock_data[:,i+look_back,1]
            next_close = test_stock_data[:,i+look_back,0]
            unext_open = unnorm(test_stock_data[:,i+look_back,1], maxx=maxo, minx=mino)
            unext_close = unnorm(test_stock_data[:,i+look_back,0], maxx=maxc, minx=minc)
        else:
            unext_open = test_stock_data[:,i+look_back,1]
            unext_close = test_stock_data[:,i+look_back,0]


        
        if predictor is not None:
            if i==0 or (i==1 and inc_based):
                old_y = x_in[:,-1,0]

            if inc_based:
                x_inv= x_in -test_stock_data[:,i-1:i+look_back-1,:]
            else:
                x_inv = x_in
            if use_nopen:
                base_val = unext_open   
            else:
                base_val=x_in[:,-1,0]
            
            if st:
                y_out = predictor(x_inv, base_val=base_val)
            else:
                y_out = predictor(x_inv, base_val=base_val, stochastic=st)

            for j in range(test_stock_data.shape[0]):
                plot_predict[j].append(y_out[j].detach().cpu())

        if strategy == "Long-only":
            for k in range(test_stock_data.shape[0]):
                if predictor is not None and predictor_mode!="coinflip":
                    if predictor_mode=="naive":
                        condition = y_out[k]>x_in[k,-1,0]
                    
                    elif predictor_mode=="noref":
                        condition = y_out[k]>old_y[k]    
                    
                    elif predictor_mode=="openaware":
                        if normalize:
                            uy_out = y_out[k]*(maxc[k]-minc[k])+minc[k]       
                        else:
                            uy_out = y_out[k]
                        condition = uy_out > unext_open[k]
                    
                    elif predictor_mode=="taxaware":
                        if normalize:
                            uy_out = y_out[k]*(maxc[k]-minc[k])+minc[k]       
                        else:
                            uy_out = y_out[k]
                        condition = (budg[k]*(1-tax))*(1+(uy_out-unext_open[k])/unext_open[k])*(1-tax) - budg[k] > 0
                    
                    elif predictor_mode=="perfect":
                        condition = unext_close[k] > unext_open[k]
                    
                    elif predictor_mode=="test":
                        condition = x_in[k,-1,0] > unext_open[k]

                    else:
                        raise Exception(f"Unknwon predictor mode \"{predictor_mode}\"  {i}{k}")
                else:
                    condition = torch.rand(1)>=0.5

                if condition:
                    o_budg = budg[k].clone()
                    budg[k] = (budg[k]*(1-tax))*(1+(unext_close[k]-unext_open[k])/unext_open[k])*(1-tax)
                    profit[k].append(budg[k]-o_budg)
                    payed[k] += o_budg
                    avg_prof_metrics[k].append(budg[k]/o_budg)
                    avg_growth[k].append(unext_close[k]/unext_open[k])
                    nor_avg_prof_met.append((budg[k]/o_budg)/(unext_close[k]/unext_open[k]))     
                else:
                    profit[k].append(torch.tensor(0.))
                    avg_prof_metrics[k].append(torch.tensor(1.))
                    nor_avg_prof_met.append(1/(unext_close[k]/unext_open[k])) 
                    avg_growth[k].append(unext_close[k]/unext_open[k])
                    

        if predictor is not None:
            old_y = y_out
    
    last_day = i+look_back

    tot = 0
    tot_payed = 0
    growth_budg = 0
    tot_actions=0
    strat_perf = []
    pred_profit = []
    if do_print:
        print(f"\nPredictor Mode:     {predictor_mode}")
        print(f"Tax fee:            {tax}")
        print(f"Invest Time:        {invest_time}")
        print(f"Starting budget:    {budg0} $")

    for i,s in enumerate(stock_dict):
        TP = torch.sum(torch.tensor(profit[i], device=test_stock_data.device)>0)
        P = torch.sum((test_stock_data[i,look_back:look_back+invest_time,0]-test_stock_data[i,look_back:look_back+invest_time,1])>0)
        FP = torch.sum(torch.tensor(profit[i], device=test_stock_data.device)<0)
        FN = torch.sum(torch.logical_xor(P,TP))

        summ = torch.sum(torch.tensor(profit[i]))
        summ_p = torch.sum(payed[i])
        avg_growth[i] = torch.mean(torch.tensor(avg_growth[i]))
        avg_prof_metrics[i] = torch.mean(torch.tensor(avg_prof_metrics[i]))/avg_growth[i]
        nor_avg_prof_met[i] = torch.mean(torch.tensor(nor_avg_prof_met[i]))
        if do_print:
            print(f"\n-TICKER [{s}]")
            print(f"Actions:         {(TP+FP)}")
            print("Avg Op-Profit:   {:.3f} $".format(summ))
            print("Payed:           {:.2f}".format(summ_p))
            if summ_p>0:
                print("Avg Op-Return:   {:.3f} %".format((summ/summ_p*100))) 
            print("Precision:       {:.3f}".format((TP/(TP+FP))))  # how many right call among model's actions
            print("Recall:          {:.3f}".format((TP/P)))  # how many right call among the whole right calls
            print("F1 score:        {:.3f}".format((TP/(TP + 0.5*(FP+FN))))) 
            pred_profit.append(budg[i]-budg0)
            print("Pred profit:     {:.2f} $    [{:.3f} %]".format((budg[i]-budg0), (budg[i]/budg0-1)*100))
            growth = (test_stock_data[i,last_day,0]-test_stock_data[i,look_back,0])/test_stock_data[i,look_back,0]
            print("Growth profit:   {:.2f} $    [{:.3f} %]".format(((growth)*budg0), (growth)*100))
            growth_budg += (1+growth)*budg0 
            print("Strat Surplus:   {:.2f} $".format(budg[i]-(1+growth)*budg0)) 
            strat_perf.append((budg[i]/budg0)/(1+growth))
            # strategy performance:  (budg[i]/budg0) / (growth_budg/budg0) = (budg[i]/budg0) /(1+growth)
            print("Strat Perf:      {:.3f}\n".format(strat_perf[-1]))
            # budget = growth_budg*strat_perf
            # profit_strat = profit_growth*strat_perf + b*(strat_perf-1)
        else:
            pred_profit.append(budg[i]-budg0)
            growth = (test_stock_data[i,last_day,0]-test_stock_data[i,look_back,0])/test_stock_data[i,look_back,0]
            growth_budg += (1+growth)*budg0 
            strat_perf.append((budg[i]/budg0)/(1+growth))

        tot += summ
        tot_payed += summ_p
        tot_actions += TP+FP

    tot_budg0 = budg0*test_stock_data.shape[0]
    tot_prof_perc = (torch.sum(budg)-tot_budg0)/tot_budg0*100

    if do_print:
        print("\nTot Start Budg:      {:.3f} $".format(tot_budg0)) 
        print(f"Tot Actions:         {tot_actions}")  
        print("Tot Avg Op-profit:   {:.3f} $".format(tot)) 
        print("Payed:               {:.2f} $".format(tot_payed)) 
        if tot_payed>0:
            print("Avg Return:          {:.3f} %".format(tot/tot_payed*100)) 
        print("Tot Profit:          {:.2f} $    [{:.3f} %]".format(torch.sum(budg)-tot_budg0, (torch.sum(budg)-tot_budg0)/tot_budg0*100))     
        print("Tot Growth Profit:   {:.2f} $    [{:.3f} %]".format(growth_budg-tot_budg0, (growth_budg/(tot_budg0)-1)*100))
        print("Tot Strat Surplus:   {:.2f} $".format(torch.sum(budg)-growth_budg))
        # strategy performance:  (sum(budg)/tot_budg0) / (growth_budg/tot_budg0) = sum(budg)/growth_budg
        print("Tot Strat Perf:      {:.3f}\n".format(torch.sum(budg)/(growth_budg)))
                                                                 

        print(f"Investiment days:    {min(test_stock_data.shape[1]-look_back, invest_time)}  [{min(test_stock_data.shape[1]-look_back, invest_time)/365} years]")
        if training:
            print(f"Starting Date:       {(list(stock_dict.values())[0]).iloc[look_back+delay]['date']}")
            print(f"Ending Date:         {(list(stock_dict.values())[0]).iloc[last_day+delay]['date']}\n\n")
        else:
            print(f"Starting Date:       {(list(stock_dict.values())[0]).iloc[split_date+delay+look_back]['date']}")
            print(f"Ending Date:         {(list(stock_dict.values())[0]).iloc[split_date+delay+last_day]['date']}\n\n")



    return strat_perf, pred_profit, plot_predict, budg, avg_prof_metrics, nor_avg_prof_met, tot_prof_perc

