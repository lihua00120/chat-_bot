# -*- coding: utf-8 -*-

from flask import Flask, request, abort
import pandas as pd
from datetime import datetime, timedelta
import os

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage


# ============================
# 事前載入資料
# ============================
URL_TOMORROW = "https://raw.githubusercontent.com/lihua00120/veg-price-analysis/main/veg_pred.csv"
URL_PRICE = "https://raw.githubusercontent.com/lihua00120/veg-price-analysis/main/veg_prices_history.csv"

# Tomorrow prediction
df_tomorrow = pd.read_csv(URL_TOMORROW ,encoding='utf-8', sep=',')
df_tomorrow = df_tomorrow[df_tomorrow["產品名稱"] != "其他"]
tomorrow_price = dict(zip(df_tomorrow['產品名稱'], df_tomorrow['預測明日菜價(元/公斤)']))

# Recipe
df_recipe = pd.read_csv("recipe.csv")

# 過去 30 天資料
df_price = pd.read_csv(URL_PRICE ,encoding='utf-8', sep=',')
df_price['交易日期'] = pd.to_datetime(df_price['交易日期']).dt.date

today = datetime.today().date()
one_month_ago = today - timedelta(days=30)

df_recent = df_price[df_price['交易日期'] >= one_month_ago]
df_recent = df_recent[~df_recent["產品名稱"].str.contains("其他")]
df_recent['產品名稱'] = df_recent['產品名稱'].str.strip().str.split().str[0]

# 計算平均價格
avg_price_dict = df_recent.groupby('產品名稱')['加權平均價(元/公斤)'].mean().to_dict()



# ============================
# Flask 啟動
# ============================
app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("jhqh1eqTNFzl6uM30HXt14IrYgRX+wZ6bxT5Uf/snr/dWl8KkQ2jmPvzdFlFcrVVUcbsVnbzAK9IdbbfeQkJcEHXbH6mDh3pZLDHaWimIAbgjVKyqzFRYH+FpjdsuYsj/FNwpBdOCn55wkrwP9ajTwdB04t89/1O/w1cDnyilFU=")
LINE_CHANNEL_SECRET = os.getenv("70a36a332d4d50977982441825bbabc6")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)



# ============================
# 主功能：訊息處理
# ============================
def handle_user_message(user_input):
    user_input = user_input.strip()

    if user_input == "明日菜價":
        diffs = []
        for veg, avg in avg_price_dict.items():
            if veg in tomorrow_price:
                diffs.append((veg, avg, tomorrow_price[veg]))

        # 低於平均的蔬菜
        under_avg = [x for x in diffs if x[2] < x[1]]

        # 選前五名
        if len(under_avg) >= 5:
            selected = sorted(under_avg, key=lambda x: x[1] - x[2], reverse=True)[:5]
        else:
            selected = sorted(diffs, key=lambda x: abs(x[1] - x[2]))[:5]

        result = " 前五名便宜蔬菜及明日預測價格：\n"
        for veg, avg, price in selected:
            result += f"{veg} → {price:.2f} 元/公斤（平均 {avg:.1f}）\n"

        # 回傳同時保存，用於下一步的「建議食譜」
        return result

    elif user_input == "建議食譜":
        result = " 依據便宜蔬菜推薦食譜：\n"

        # 從便宜菜中挑前幾名
        cheap_veggies = sorted(
            avg_price_dict.items(), key=lambda x: x[1]
        )[:5]  # avg 低的前五名

        for veg, _ in cheap_veggies:
            recipes = df_recipe[
                df_recipe["主要食材"].str.contains(veg, na=False)
            ]

            if recipes.empty:
                result += f"❌ {veg} 找不到食譜\n"
            else:
                result += f"\n🟦 {veg} 可料理：\n"
                for _, row in recipes.iterrows():
                    result += f"- {row['菜名']}（主食材：{row['主要食材']}）\n"

        return result

    else:
        return "請輸入以下指令：\n1️⃣ 明日菜價\n2️⃣ 建議食譜"



# ============================
# Webhook 入口
# ============================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


# ============================
# LINE 訊息事件
# ============================
@handler.add(MessageEvent, message=TextMessage)
def message_event(event):
    user_text = event.message.text
    reply_text = handle_user_message(user_text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )



# ============================
# 主程式啟動（給 Render）
# ============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


"""#分詞 以防有人多打"""

# 檢視資料
#df.head(10)

# import jieba

# def cutProcess(sting):
#     result = jieba.lcut(sting)
#     result = " ".join(result)

#     return result

# df['quote'] = df['quote'].apply(cutProcess)

# df.head(5)

"""#訓練資料"""

# data = df
# training_documents = data['quote'].values.astype('U')
# labels = data['category'].values.astype('U')

# #切分训练集和测试集，分为80%训练集，20%测试集
# X_train, X_test, y_train, y_test = train_test_split(training_documents, labels, test_size=0.1, random_state=12)


# vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b') # token_pattern='(?u)\\b\\w+\\b' 單字也計入
# x_train = vectorizer.fit_transform(X_train)

# # train
# classifier = MultinomialNB (alpha=0.01) # change model type here
# classifier.fit(x_train, y_train)

# training_documents

# for i in x_train:
#     print("#:"i)

"""#評估模型

"""

# x_test = vectorizer.transform(X_test)
# classifier.score(x_test,y_test)

# print(X_test)
# predict(X_test)

# def predict(raw_queries,n_top=1):
#     raw_queries = [cutProcess(s) for s in raw_queries]
# #     print(raw_queries)

#     queries = vectorizer.transform(raw_queries)
#     predict =  classifier.predict_proba(queries).tolist()
#     predict = [{k:round(v,4) for k,v in zip(classifier.classes_[:3],qa[:3])} for qa in predict]
#     predict = [ sorted(dictt.items(), key=lambda d: d[1], reverse=True) for dictt in predict]
#     return predict

# example = ['我有問題','修改公司資料','我想在台中市東山路附近找小雞上工上的工作','要怎麼變更公司電話','您好應徵者為何看不到我們需要出差的項目']

# lists = predict(example)

# for index,qa in enumerate(lists):
#     print("question:",example[index])
#     print("anser:", qa)

#     print()

# txt = input()
# predict([txt])[0]

