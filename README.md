# IOT_final


我的分心辨識儀能夠輕鬆辨識再網路攝影機前的你現在正在做甚麼，主要是利用Openpose來製作完成，首先是利用my_pose_sampler這個python程式來錄製樣本，它會將你的身體各個關鍵點位置找出來，同它會創建一個文件夾，裡面有你拍的每一張照片(2秒拍攝一張)，並且將你所有照片中身體關鍵點的位置紀錄在一個Csv檔案中，這時利用
