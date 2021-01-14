# IOT_final


我的分心辨識儀能夠輕鬆辨識再網路攝影機前的你現在正在做甚麼，主要是利用Openpose來製作完成，首先是利用my_pose_sampler這個python程式來錄製樣本，它會將你的身體各個關鍵點位置找出來，同它會創建一個文件夾，裡面有你拍的每一張照片(2秒拍攝一張)，並且將你所有照片中身體關鍵點的位置紀錄在一個Csv檔案中，這時利用Distract_detecter這個檔案來將收集到的資料放入ai模型之中訓練，出來的結果會成為一個pkl檔案，將此檔案(detecter3.pkl)放入我們的最終辨識器distract_detecter_ver1並利用Anaconda環境的控制台執行它，就能準確地對三種正在進行的行為進行辨識(讀書、睡覺、滑手機)。

首先從安裝好anaconda開始，前往 https://www.anaconda.com/products/individual  
去下載anaconda，
詳細的安裝方法請參考  https://medium.com/python4u/anaconda%E4%BB%8B%E7%B4%B9%E5%8F%8A%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-f7dae6454ab6

安裝好anaconda後，在anaconda中安裝進我們所有需要的package
在Enviroment中的Search Package欄位中安裝下列package
![](https://i.imgur.com/p7G9lV0.png)

numpy  
matplotlib  
seaborn  
pandas  
keras  
sklearn  
tensorflow  
joblib  

安裝都好了之後開始安裝openpose，這將會是個大工程，由於openpose非常吃效能，請各位務必確保自己的電腦有高性能顯示卡與記憶卡(至少GTX1050ti，Ram8GB以上)，首先，請參考下列網址:  
https://kknews.cc/zh-tw/code/ab9z2q6.html  
openpose：https://github.com/CMU-Perceptual-Computing-Lab/openpose
（3rd party裡的caffe和pybind11要另外載再放進資料夾裡，由於有些人安裝openpose時會發生caffe和pybind11資料夾為空的情況，下載好openpose請依照情況判斷要不要安裝）  
 -  caffe：https://github.com/CMU-Perceptual-Computing-Lab/caffe/tree/b5ede488952e40861e84e51a9f9fd8fe2395cc8a
 - pybind11：https://github.com/pybind/pybind11/tree/085a29436a8c472caaaf7157aa644b571079bcaa  
接下來是安裝visual stuidio2015，由於原本openpose是用C來完成的，要給Python使用需要另外編譯，這時候就需要VS，版本不一定要2015後都可，安裝載點如下:  
https://visualstudio.microsoft.com/zh-hant/vs/older-downloads/  
接著是Cmake，用來建路徑:  
https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork  
接著是CUDA，它是Nvidia用來給GPU進行平行運算的框架，所以只能使用N家顯卡，一樣先下載CUDA v11.1，(安裝請務必要再VS裝好之後，因為CUDA安裝會產生Visual Stuido需要的檔案):  
：https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork  
再來是CuDNN，它是Nivida給神經網路用的GPU加速庫，這裡用cuDNN v8.0.4 for CUDA 11.1:  
https://developer.nvidia.com/rdp/cudnn-archive  
有些人可能沒有Python，這裡提供python v2.7.18:  
https://www.python.org/downloads/release/python-2718/  
以及opencv-python v3.2.08，複製裡面的程式碼放到CMD中下載:  
https://pypi.org/project/opencv-python/3.2.0.8/  

全部好了之後打開CMAKE開始編譯需要的檔案  
![](https://i.imgur.com/LoUPvkY.png)  
設定檔案路徑（在 clone 下來的路徑中自己創一個 build 資料夾，並設定成 Where to build the binaries ）    
![](https://i.imgur.com/bMTOiHB.png)  
![](https://i.imgur.com/vtpteZZ.png)  
按下 Configure 按鈕（generator 記得選你的VS版本）  
![](https://i.imgur.com/TL7PTYW.png)  
等待 Configure Done  
![](https://i.imgur.com/ldUXJr7.png)
勾選 BUILD_PYTHON，按下Generate  
完成之後用 Visual Studio 2019打開build/OpenPose.sln檔案，
切換到Release Mode並Build Project(一定要用release mode，只用debug會失敗)  
如果上面的步驟都做完，應該可以在 openpose\build\python\openpose\Release 看到 openpose_python.cp36-win_amd64.pyd library 文件。  
這樣openpose大致安裝完畢了，開始錄製判斷分心ai所需要的樣本資料。  
在你放置openpose的資料夾(我的是在C:\Users\allen\Documents\GitHub\openpose，依據安裝位置可能不一樣)，中尋找你之前CMAKE編譯時創的build資料夾，把my_pose_sampler放進去，並在cmd中打py my_pose_sampler來執行，順利的話它會顯示視窗如下，並在在同一個檔案目錄中創造一個資料夾，裡面有每隔兩秒拍攝的照片以及一個csv檔案。
![](https://i.imgur.com/dSgEzlI.jpg)  
如果openpose安裝成功，視窗會像上面那樣  
![](https://i.imgur.com/Bf9SkjY.png)  
資料名稱是my_sample_(現在時間)  
![](https://i.imgur.com/jF7c8gU.png)  
資料夾內如上  
製作樣本的方法如下:  
1.打開你的my_pose_sampler，我是使用Notepad++  
2.在第87行的位置「csvWriter.writerow([0,」這行請依據你現在的樣本模式填入0、1、2的數字(我是0代表讀書，1代表睡覺，2代表滑手機)   
![](https://i.imgur.com/2thUf1V.png)  
3.這時候輸出檔案的csv的y(認真狀態)就會依照你的樣本被賦予編號了，將三個csv檔案合在一起，(建議直接用Excel打開csv檔案將下方數據部分複製貼上，省時省力  
![](https://i.imgur.com/dTQ3lgu.png)
4.將他們全部存到同一個csv檔案中，因為y已經寫入了當時的讀書狀態所以不用特別在意順序  

完成樣本之後，現在開始利用蒐集到的資訊訓練模型，打開anaconda並打開jupyter notebook，放入Distract_detecter.ipynb檔案，將你剛剛蒐集好的csv，命名為sample_coords.csv(若是正常當初自動創建的csv名稱就是這樣不需要改變)，並放入和前述檔案相同的資料夾，接著打開Distract_detecter.ipynb後執行它(需要的package都已經安裝完畢，若仍有需要請確認是否都有安裝)，跑過一段時間後，它會輸出一個名為detecter3.pkl的pkl檔案在同一個資料夾，這就是我們需要的判斷ai。  
![](https://i.imgur.com/QCNPOpF.png)
將ai放置回到openpose的build資料夾底下，同時也將及時判斷程式distract_detecter_ver1放入build資料夾，接著使用anaconda打開cmd(因為程式需要的package在anaconda中，當然若在window中安裝過上述package可以直接打開cmd)，打入cd (build資料夾所在的路徑)，之後打python distract_detecter_ver1.py來執行它(切記要打python，單單打py會報錯，my_pose_sampler則不需要也沒關係)  
![](https://i.imgur.com/MSAseYy.png)


接著他就會將你訓練好的pkl檔案引入，開啟視窗並顯示你現在即時的狀態，由於是兩秒偵測一次所以要稍等一下  
![](https://i.imgur.com/XL6xNRr.png)





此外若出現Cuda記憶體不夠的報錯請將chrome關掉(非常有用，若還是不行可以在執行的時候打 "--net_resolution -1x224"來透過降低判斷準確度換取GPU順暢度，這樣還是不行請升級顯卡吧)，實測影片我放在檔案中了，可以參考。




