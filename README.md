1.Create a twitter account
2.Create a accounts.txt which contains your account information -- username:password:email:password (format)
3.run -- twscrape add_accounts accounts.txt username:password:email:password
4.twscrape login
5.py app.py

Description:

app.py - contains the flask application code along with the model for training on the previous 90d data fetched from yfinance, and tweets scraped using twscrape.
          saves scaler.pkl and model.pkl, which retrains everyday with new data.

Stock market prediction using Sentiment.ipynb - used for testing model against yfinance data and tweets data downloaded from kaggle. The model is trained and tested on tweets
                                                and yfinance data, and tested, and the same model is them implemented on the real-time data fetched from previous 90d in the 
                                                actual app.
