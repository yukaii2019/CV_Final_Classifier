# Model 相關檔案
* module.py: 放model資料夾
* trainer.py: 改裡面用的model
* inference.py: 改裡面用的model

# Preprocessing 相關檔案
* data.py: 裡面有標註的地方拿掉或放上去
* inference.py: 裡面有標註的地方拿掉或放上去

# Data augmentation 相關檔案
* data.py: 裡面有標註的地方拿掉或放上去

# 跟 imbalance 相關的部分，可以試試看的
* trainer.py: 裡面的 cross entropy 目前是有加 weight 比例是9:1是因為我看data裡面張眼閉眼的比例是9:1，這個也可以試試看拿掉或裝回來

# 其他說明：
* 目前用來做validation的是每個sequence的後面20%，跟之前寫的有點不一樣，想調比例可以從data.py裡面找
* trainer目前是用validation auc來選存下來的Module，或許可以從其他指標來選，但我覺得應該不能是atnr，因為全猜0的model會拿很高分
* 新增的一個conf.json是存training data張眼閉眼的答案的，要喬一下.sh裡面的路徑
* 我這邊都沒有真的train滿 200 epoch 大概都跑20幾個而已，有時候放一個晚上可能會跑很多，可能可以憑感覺?

# train
```shell script=
bash run_train.sh
```

# inference
```shell script=
bash run_inference.sh
```

