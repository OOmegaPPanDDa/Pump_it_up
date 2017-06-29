Prepare
1. 將資料放在 test/data 資料夾
2. Run test/download_model.sh 並解壓縮且將資料夾，命名成 test/save

Run Order

(preprocess part)
1. use python3 run test/preprocess.py
2. use python3 run test/location_pred.py

(test part)
3. use python3 run test/test.py
4. use python3 run test/format.py
5. test/test_label.csv 為最後檔案

Run Script (in test file)
1.	Run test/doall.sh