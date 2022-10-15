import numpy as np
from paramiko import PKey
import src.pl_data.correlation as corr
import pandas as pd
import torch
import yfinance as yf
import pickle as pk
from src.common.utils import PROJECT_ROOT

def setup_stock(start_date, stock_df, ref, flat_stocks = False, normalize = False, polish_n = 0):

    min_open = np.min(stock_df[ref])
    max_open = np.max(stock_df[ref])
    init_stock_value = stock_df[ref][0]
    dates = corr.compute_dates_vec(stock_df.index)

    strend_pday = corr.fill_trend(dates, stock_df[ref])
    if flat_stocks:
        strend_pday = corr.flatter(strend_pday)

    tmp = [i for i in range(dates[0], len(strend_pday)+dates[0])]
    dates_fill = []
    for d in tmp: dates_fill.append(corr.to_date(start_date, d))

    strend_pday = pd.DataFrame(
        strend_pday,
        columns=[ref]
    )

    if polish_n>0:
        strend_pday["Unpolished ST Data"] = strend_pday[ref]
    strend_pday.index = dates_fill


    if flat_stocks:
        min_open_flat = min(strend_pday[ref])
        max_open_flat = max(strend_pday[ref])
        if normalize:
            strend_pday = (strend_pday-min_open_flat)/(max_open_flat-min_open_flat)
    else:
        if normalize:
            strend_pday = (strend_pday-min_open)/(max_open-min_open) 

    if polish_n>0:
        strend_pday[ref] = corr.polish(strend_pday[ref], polish_n)
    
    strend_pday.reset_index(inplace=True)
    strend_pday.rename(columns = {'index':'date'}, inplace = True)
    
    return strend_pday


def setup_stock_vec(stock_df, flat_stocks, normalize, polish_n, line):

    s_list = []
    start_date = str(stock_df.index[0])
    for i, ref in enumerate(stock_df.columns):
        min_open = np.min(stock_df[ref])
        max_open = np.max(stock_df[ref])
        init_stock_value = stock_df[ref][0]
        dates = corr.compute_dates_vec(stock_df.index)

        strend_pday = corr.fill_trend(dates, stock_df[ref], line=line[i])
        if flat_stocks[i]:
            strend_pday = corr.flatter(strend_pday)

        tmp = [i for i in range(dates[0], len(strend_pday)+dates[0])]
        dates_fill = []
        for d in tmp: dates_fill.append(corr.to_date(start_date, d))

        strend_pday = pd.DataFrame(
            strend_pday,
            columns=[ref]
        )

        if polish_n[i]>0:
            strend_pday[f"Unpolished {ref}"] = strend_pday[ref]
        strend_pday.index = dates_fill


        if flat_stocks[i]:
            min_open_flat = min(strend_pday[ref])
            max_open_flat = max(strend_pday[ref])
            if normalize[i]:
                strend_pday = (strend_pday-min_open_flat)/(max_open_flat-min_open_flat)
        else:
            if normalize[i]:
                strend_pday = (strend_pday-min_open)/(max_open-min_open) 

        if polish_n[i]>0:
            strend_pday[ref] = corr.polish(strend_pday[ref], polish_n[i])
        

        s_list.append(strend_pday)

    strend_pday = pd.concat(s_list, axis=1)
    strend_pday.reset_index(inplace=True)
    strend_pday.rename(columns = {'index':'date'}, inplace = True)
  
    return strend_pday





def get_dataset(device="cpu"):


    datapath = "C:/Users/mvisc/Desktop/Appunti/Deep Learning/progetto/S-LSTM-Stock-Prices-Forecasting/data/stock_trends/worked/"
    raw_datapath = "C:/Users/mvisc/Desktop/Appunti/Deep Learning/progetto/S-LSTM-Stock-Prices-Forecasting/data/stock_trends/raw/"
    fac_datapath = "C:/Users/mvisc/Desktop/Appunti/Deep Learning/progetto/S-LSTM-Stock-Prices-Forecasting/data/stock_trends/factors/worked/"



    datatypes = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
        'Stock Splits'] + ['Date', 'Quarterly Revenue', 'TTM Net Margin', 'TTM Operating Margin',
        'TTM Gross Margin', 'Quarterly EPS', 'Long Term Debt',
        'Shareholder\'s Equity_x', 'Debt to Equity Ratio', 'TTM Net Income',
        'Shareholder\'s Equity_y', 'Return on Equity', 'Stock Price',
        'Book Value per Share', 'Price to Book Ratio']

    #select_vec = ["Close", "Volume", "Debt to Equity Ratio", "Return on Equity", "Price to Book Ratio", "TTM Net Margin", "Quarterly EPS"]   #beta
    select_vec = ["Close", "Open", "Volume", "Debt to Equity Ratio", "Return on Equity", "Price to Book Ratio", "TTM Net Margin", "Quarterly EPS"]   #beta
    #select_vec = ["Close", "Open", 'High', 'Low', 'Volume']




    #Apple, Amazon, berkshire hathaway, Google, J&J, Microsoft, Tesla, Meta
    symbols = ["AAPL", "AMZN", "BRK-B", "GOOGL", "JNJ", "MSFT", "TSLA", "MMM", "BEN"]#, "META"]
    symbols = symbols[:4]
    symbols = ["AAPL"]
    #symbols = ["GOOGL"]

    start_date = '2011-03-31' #None #'2011-03-31' #'2010-5-31'
    end_date = None #'2022-04-01' #'2020-5-31'
    period = 'max'
    flat_stocks = False
    normalize = False
    polish_n = 0
    line_interp = True
    max_shift = 20
    ref = "Close" #"Open" 
    get_raw_online = False
    rework_data = True
    crop_to_common_dates = True

    norm_list=[
        False, False, False, False, True, False, False
    ]
    flat_list=[
        flat_stocks, flat_stocks, flat_stocks, flat_stocks, flat_stocks, False, False
    ]
    polish_list=[
        polish_n, polish_n, polish_n, polish_n, polish_n, 0, 0
    ]
    line_list=[
        line_interp, line_interp, line_interp, line_interp, line_interp, False, False
    ]


    #print(f"Retrieving Stocks data\n")
    stock_dict = {}
    missing = []
    factors_dict = {}
    for i, s in enumerate(symbols):
    #for i in trange(len(symbols)):
    #    s = symbols[i]
        p = round(i/len(symbols),3)*100
        pstr = str(p)[:min(len(str(p)), 5)]
        print(f"[{pstr}%] Retrieving data for {s} ..." +" "*(25-len(s)-len(pstr)) + "■"*int(p/10)+"□"*int(10-p/10), end="\r")
        # ☐ ▧ ■ □
        
        if rework_data:
            if get_raw_online:
                stock_Data = yf.Ticker(s)
                # Open  High    Low Close   Volume  Dividends   Stock Splits
                if start_date is not None or end_date is not None:
                    stock_df = stock_Data.history(start=start_date, end=end_date, auto_adjust=True)
                else:
                    stock_df = stock_Data.history(period="max", auto_adjust=True)
            else:
                stock_df = pd.read_json(raw_datapath + f"{s}.json", orient="index")
            
            if start_date is not None or end_date is not None:
                mask = True
                if start_date is not None:
                    mask = mask & (stock_df.index>= pd.to_datetime(start_date))
                if end_date is not None: 
                    mask = mask & (stock_df.index <= pd.to_datetime(end_date))
                
                stock_df = stock_df.loc[mask]
                

            if stock_df.empty:
                #raise Exception("**This ticker do not exists!**")
                missing.append(s)
                continue

            #stock_pday = setup_stock(start_date, stock_df, ref, normalize=normalize, polish_n=polish_n, flat_stocks=flat_stocks)
            stock_pday = setup_stock_vec(stock_df, normalize=norm_list, polish_n=polish_list, flat_stocks=flat_list, line=line_list)
        
        else:
            stock_pday = pd.read_json(datapath + f"{s}.json", orient="index")
            if start_date is not None or end_date is not None:
                mask = True
                if start_date is not None:
                    mask = mask & (stock_pday["date"]>= pd.to_datetime(start_date))
                if end_date is not None: 
                    mask = mask & (stock_pday["date"] <= pd.to_datetime(end_date))
                stock_pday = stock_pday.loc[mask]
                stock_pday.reset_index(inplace=True, drop=True)



        # TODO: EXPERIMENTAL
        factors_pday = pd.read_json(fac_datapath+f"{s}.json", orient="index")
        factors_dict[s] = factors_pday

        #print(factors_pday)
        stock_pday = stock_pday.merge(factors_pday, how="inner")[["date"]+select_vec]

        # TODO: GTREND
        # gtrend = setup_stock(start_date,stock_pday.set_index("date"), ref, normalize=normalize, polish_n=100, flat_stocks=True)
        # gtrend = gtrend.loc[:,["date","Close"]]#.set_index("date")  #gtrend.drop(1, axis="columns")
        # gtrend.rename(columns={"Close":"Advisor"}, inplace=True)
        # stock_pday = stock_pday.merge(gtrend, how="inner")

                

        # Close Open  High  Low  Volume  Dividends   Stock Splits
        close_col = stock_pday.pop('Close')
        stock_pday.insert(1, 'Close', close_col)

        if len(stock_dict)>0 and not crop_to_common_dates:
            if len(stock_pday)==len(list(stock_dict.values())[-1]):
                stock_dict[s] = stock_pday
            else:
                print(f"\nDISCARDING Ticker {s} for unmatching dimensions [{len(stock_pday)}]")
        else:
            stock_dict[s] = stock_pday


    #date  Close   Open   Volume  Debt to Equity Ratio  Return on Equity  Price to Book Ratio  TTM Net Margin  Quarterly EPS




    print("* Stock-data retrieval completed *                                                             ")
    print(f"Missing stocks: {missing}")



    if crop_to_common_dates:
        min_date = stock_dict[list(stock_dict.keys())[0]]["date"].iloc[0]
        max_date = stock_dict[list(stock_dict.keys())[0]]["date"].iloc[-1]
        for s in stock_dict:
            if min_date < stock_dict[s]["date"].iloc[0]:
                min_date = stock_dict[s]["date"].iloc[0]
            if max_date > stock_dict[s]["date"].iloc[-1]:
                max_date = stock_dict[s]["date"].iloc[-1]
        for s in stock_dict:   
            mask = (stock_dict[s]["date"]>= min_date) & (stock_dict[s]["date"]<= max_date)
            stock_dict[s] = stock_dict[s].loc[mask]
            stock_dict[s].reset_index(inplace=True, drop=True)


    tensor_stock_dict= {}
    for i,s in enumerate(stock_dict):
        stk = stock_dict[s]
        tensor_stock_dict[s] = torch.concat([torch.tensor(stk[c], device=device).reshape(1,-1) for c in stk.columns[1:]],dim=0).float()
        

    stock_data = torch.concat([tensor_stock_dict[s].unsqueeze(0) for s in tensor_stock_dict], dim=0)
    stock_data = stock_data.transpose(1,2)   # tickers, seq length, data dim


    # Data I'm going to consider
    # Close Open High Low Volume Dividends Stock Splits
    #stock_data = stock_data[:,:,:5]
    #stock_data = stock_data[:,:,0].unsqueeze(2)


    if normalize:
        maxc = torch.zeros(stock_data.shape[0])
        minc = torch.zeros(stock_data.shape[0])
        maxo = torch.zeros(stock_data.shape[0])
        mino = torch.zeros(stock_data.shape[0])
        for j in range(stock_data.shape[0]): 
            for k in range(stock_data.shape[2]):
                if k==0:
                    maxc[j] = torch.max(stock_data[j,:,k]) 
                    minc[j] = torch.min(stock_data[j,:,k])
                if k==1:
                    maxo[j] = torch.max(stock_data[j,:,k]) 
                    mino[j] = torch.min(stock_data[j,:,k])
                stock_data[j,:,k] = (stock_data[j,:,k]-torch.min(stock_data[j,:,k]))/(torch.max(stock_data[j,:,k])-torch.min(stock_data[j,:,k]))


    # Adding next day open price as training input
    # stock_data[:,:,1] must be the open price 
    #stock_data = torch.concat((stock_data[:,:-1,:],stock_data[:,1:,1].unsqueeze(2)),dim=2)


    train_data_perc = 0.7
    renormalize = False

    split_date = int(stock_data.shape[1]*train_data_perc)
    train_stock_data = stock_data[:,:split_date,:]
    test_stock_data = stock_data[:,split_date:,:]



    #Renormalization
    if renormalize:
        for j in range(train_stock_data.shape[0]):
            for k in range(train_stock_data.shape[2]):
                train_stock_data[j,:,k] = (train_stock_data[j,:,k]-torch.min(train_stock_data[j,:,k]))/(torch.max(train_stock_data[j,:,k])-torch.min(train_stock_data[j,:,k]))
        for j in range(test_stock_data.shape[0]):
            for k in range(test_stock_data.shape[2]):
                test_stock_data[j,:,k] = (test_stock_data[j,:,k]-torch.min(test_stock_data[j,:,k]))/(torch.max(test_stock_data[j,:,k])-torch.min(test_stock_data[j,:,k]))


    splitvaltest = test_stock_data.shape[1]//3

    print(f"\nStock retrieved: {stock_data.shape[0]}")
    print(f"Training data percentage: {train_data_perc}")
    print(f"Training data len: {train_stock_data.shape[1]}")
    print(f"Test data len: {test_stock_data.shape[1]}")
    print(f"Data shape: {stock_data.shape}")
    print(f"Starting Training Date: {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[0]}")
    print(f"Ending Training Date:   {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[split_date-1]}")
    print(f"Starting Val Date:  {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[split_date]}")
    print(f"Ending Val Date:    {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[split_date+splitvaltest]}")
    print(f"Starting Testing Date:  {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[split_date+splitvaltest]}")
    print(f"Ending Testing Date:    {stock_dict[list(stock_dict.keys())[0]]['date'].iloc[-1]}\n\n")


    dataset ={

        "StockDict": stock_dict,

        "train": train_stock_data,#.transpose(0,1),

        "test": test_stock_data[:,splitvaltest:,:], #.transpose(0,1) 

        "val": test_stock_data[:,:splitvaltest,:]
    }

    return dataset


if __name__ == "__main__":
    data = get_dataset()

    print(PROJECT_ROOT)

    with open(str(PROJECT_ROOT)+'/data/stock_dict.pickle', 'wb') as pfile:
        pk.dump({"train": data["train"]}, pfile)

    with open(str(PROJECT_ROOT)+'/data/train/train_dataset.pickle', 'wb') as pfile:
        pk.dump({"train": data["train"]}, pfile)

    with open(str(PROJECT_ROOT)+'/data/test/test_dataset.pickle', 'wb') as pfile:
        pk.dump({"test": data["test"]}, pfile)

    with open(str(PROJECT_ROOT)+'/data/val/val_dataset.pickle', 'wb') as pfile:
        pk.dump({"val": data["val"]}, pfile)

