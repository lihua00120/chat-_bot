# -*- coding: utf-8 -*-

from flask import Flask, request, abort
import pandas as pd
from datetime import datetime, timedelta
import os
import re
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import FlexSendMessage,MessageEvent,MessageAction, TextMessage, TextSendMessage


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

#================
#換名字
name_map = {
    "青花菜": "青花苔",
    "青花苔": "青花苔",
    "青江白菜": "青江白菜",
    "小白菜": "青江白菜",
    "隼人瓜": "隼人瓜",
    "佛手瓜": "隼人瓜",
    "薯蕷": "薯蕷",
    "山藥": "薯蕷",
    "蕹菜": "蕹菜",
    "空心菜": "蕹菜",
    "萊豆": "萊豆",
    "蠶豆": "萊豆",
    "花椰菜": "花椰菜",
    "白花椰": "花椰菜",
    "胡瓜": "胡瓜",
    "小黃瓜": "胡瓜",
    "甘藷": "甘藷",
    "地瓜": "甘藷",
    "甘藍": "甘藍",
    "高麗菜": "甘藍", 
    "球莖甘藍": "球莖甘藍",  # 如果 CSV 裡叫球莖甘藍就保留
    "敏豆": "敏豆",
    "四季豆": "敏豆",
    "扁蒲": "扁蒲",
    "蒲瓜": "扁蒲",
    "芋": "芋",
    "芋頭": "芋",
    "濕香菇": "濕香菇",
    "香菇": "濕香菇",
    "濕木耳": "濕木耳",
    "木耳": "濕木耳",
    "落花生": "落花生",
    "花生": "落花生",
    "黃秋葵": "黃秋葵",
    "秋葵": "黃秋葵",
    "青蔥": "青蔥",
    "蔥": "青蔥",
    "萵苣菜": "萵苣菜",
    "A菜": "萵苣菜",
    "芫荽": "芫荽",
    "香菜": "芫荽",
    "甘藷葉": "甘藷葉",
    "地瓜葉": "甘藷葉"
}


display_map = {
    "青花苔": "青花菜",
    "青江白菜": "小白菜",
    "隼人瓜": "佛手瓜",
    "薯蕷": "山藥",
    "蕹菜": "空心菜",
    "萊豆": "蠶豆",
    "花椰菜": "白花椰",
    "胡瓜": "小黃瓜",
    "甘藷": "地瓜",
    "甘藍": "高麗菜",
    "球莖甘藍": "高麗菜",
    "敏豆": "四季豆",
    "扁蒲": "蒲瓜",
    "芋":"芋頭",
    "濕香菇":"香菇",
    "濕木耳":"木耳",
    "落花生":"花生",
    "黃秋葵":"秋葵",
    "青蔥":"蔥",
    "萵苣菜":"A菜",
    "芫荽":"香菜",
    "甘藷葉":"地瓜葉"
}


# ============================
# Flask 啟動
# ============================
app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

#Flex Recipe Bubble 模板
def make_recipe_bubble(row, default_img, veg_display=None):
    return {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": row.get("圖片網址", default_img),
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": row.get("菜名", f"{veg_display} 找不到食譜"),
                    "weight": "bold",
                    "size": "lg"
                },
                {
                    "type": "text",
                    "text": (
                        f"主食材：{row.get('主要食材','')}\n"
                        f"輔助食材：{row.get('輔助食材','')}\n"
                        f"熱量：{row.get('熱量 kcal','')} kcal\n"
                        f"蛋白質：{row.get('蛋白質 g','')} g\n"
                        f"碳水：{row.get('碳水 g','')} g"
                    )[:120],
                    "wrap": True,
                    "size": "sm",
                    "color": "#555555"
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {
                    "type": "button",
                    "action": {
                        "type": "message",
                        "label": "返回",
                        "text": "明日菜價"
                    },
                    "style": "primary",
                    "height": "sm"
                }
            ]
        }
    }


# ============================
# 主功能：訊息處理
# ============================
def handle_user_message(user_input):
    user_input = user_input.strip()

    # 共用：取得跌價蔬菜前五名
    def get_top5_cheapest():
        diffs = []
        for veg, avg in avg_price_dict.items():
            if veg in tomorrow_price:
                pred = tomorrow_price[veg]
                diff = avg - pred  # 正值 = 比平均便宜
                diffs.append((veg, avg, pred, diff))  # (菜名, 月均, 預測, 差值)

        under_avg = [d for d in diffs if d[3] > 0]
        # 跌幅從大到小排序
        if len(under_avg) >= 5:
            return sorted(under_avg, key=lambda x: x[3], reverse=True)[:5]

        return sorted(diffs, key=lambda x: abs(x[3]))[:5]

    def find_recipes(vegs):
        bubbles = []
        default_img = "https://raw.githubusercontent.com/lihua00120/chat-_bot/refs/heads/main/images/%E4%B8%89%E6%9D%AF%E8%A0%94%E8%8F%87.jpg"
        
        for veg in vegs:
            veg_search = name_map.get(veg, veg)          # 查食譜用
            veg_display = display_map.get(veg_search, veg_search)  # 顯示用
            
            recipes = df_recipe[
                df_recipe["主要食材"].str.contains(veg_search, na=False)|
                df_recipe["輔助食材"].str.contains(veg_search, na=False)
            ]
            if recipes.empty:
                bubble = {
                    "type": "bubble",
                    "hero": {
                        "type": "image",
                        "url": default_img,
                        "size": "full",
                        "aspectMode": "cover"
                    },
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "text",
                                "text": f"{veg_display} 找不到食譜",
                                "weight": "bold",
                                "size": "lg"
                            },
                            {
                                "type": "text",
                                "text": "暫無建議菜單",
                                "size": "sm",
                                "wrap": True
                            }
                        ]
                    },
                    "footer": {  # 🌟 加上返回按鈕
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                       "type": "message",
                                        "label": "返回",
                                        "text": "明日菜價"
                                },
                                "style": "primary",
                                "height": "sm"
                            }
                        ]
                    }
                }
                bubbles.append(bubble)
            else:
                for _, row in recipes.iterrows():
                    bubble = make_recipe_bubble(row, default_img)
                    bubbles.append(bubble)

        return bubbles
        
    if user_input == "明日菜價":
        
        selected = get_top5_cheapest()

        if not selected:
                return TextSendMessage("⚠️ 明日沒有任何蔬菜低於月平均價！")

        result = " 前五名便宜蔬菜及明日預測價格：\n"
        for veg, avg, price, diff in selected:
            veg_display = name_map.get(veg, veg)
            result += f"{veg_display} → {price:.2f} 元/公斤（比月均低 {diff:.1f}）\n"

        return TextSendMessage(result)


    elif user_input == "建議食譜":
        selected = get_top5_cheapest()
        vegs = [veg for veg, avg, pred, diff in selected]
        bubbles = find_recipes(vegs)
        return FlexSendMessage(
            alt_text="今日便宜蔬菜建議食譜",
            contents={
                "type": "carousel",
                "contents": bubbles[:10]
            }
        )

    else:
        # 可以支援多個菜名，用逗號或空格分隔
        vegs = re.split(r"[,、 ]+", user_input)
        bubbles = find_recipes(vegs)
        if not bubbles:
             return TextSendMessage(f"❌ 找不到包含 {user_input} 的食譜")
        alt_text = f"{user_input} 食譜" if user_input.strip() else "建議食譜"
        return FlexSendMessage(
             alt_text=f"{user_input} 食譜",
             contents={
                "type": "carousel",
                "contents": bubbles[:10]
            }
        )



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
    reply_obj = handle_user_message(event.message.text)
    line_bot_api.reply_message(event.reply_token, reply_obj)



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

