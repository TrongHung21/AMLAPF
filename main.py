import sys
from PyQt5 import  QtCore, QtWidgets, uic

import pandas as pd
import sys
from PyQt5 import  QtWidgets, uic
import numpy as np
from sklearn import preprocessing
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class PageMain(QtWidgets.QMainWindow):

    def predict(self, WS, CC, RH, MT, WD):
       data = pd.read_csv('dataAI.csv', header=0)
       data = data.dropna()

       data_final = data
       data_final.columns.values

       X = data_final.loc[:, data_final.columns != 'AQI']
       y = data_final.loc[:, data_final.columns == 'AQI']


       os = SMOTE(random_state=0)
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
       columns = X_train.columns
       os_data_X, os_data_y = os.fit_resample(X_train, y_train)
       os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
       os_data_y = pd.DataFrame(data=os_data_y, columns=['AQI'])

       cols = ['WindSpeed', 'CloudCover',
               'RelativeHumidity', 'Minimum Temperature', 'Wind Direction']
       X = os_data_X[cols]
       y = os_data_y['AQI']


       logit_model = sm.Logit(y, X)
       result = logit_model.fit()

       logit_model = sm.Logit(y, X)
       result = logit_model.fit()


       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
       logreg = LogisticRegression()
       logreg.fit(X_train, y_train)



       X_test1 = pd.DataFrame([[(WS), (CC), (RH), (MT), (WD)]],
                              columns=['WindSpeed', 'CloudCover', 'RelativeHumidity', 'Minimum Temperature',
                                       'Wind Direction'])
       y_pred = logreg.predict(X_test1)


       if (y_pred == 1):
              return 1
       return 0

    def __init__(self):
        super(PageMain, self).__init__()
        uic.loadUi("AI.ui", self)

        self.btn.clicked.connect(self.setText)

        self.show()

    def setText(self):
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        e = 0.0

        WS = self.ws.text()
        CC = self.cc.text()
        RH = self.rh.text()
        MT = self.mt.text()
        WD = self.wd.text()

        if (WS == "" or CC == "" or RH == "" or MT == "" or WD == ""):
            QtWidgets.QMessageBox.warning(self, 'Thông báo', 'KHÔNG ĐƯỢC BỎ TRỐNG!')
            return
        else:
            if (WS.isdigit() == False) or (CC.isdigit() == False) or (RH.isdigit() == False) or (MT.isdigit() == False) or (WD.isdigit() == False):
                QtWidgets.QMessageBox.warning(self, 'Thông báo', 'NHẬP SAI VUI LÒNG NHẬP LẠI!')
                return

            a = a + float(WS)
            b = b + float(CC)
            c = c + float(RH)
            d = d + float(MT)
            e = e + float(WD)

            if (0 <= a <= 100) and (0 <= b <= 100) and (0 <= c <= 100) and (0 <= d <= 100) and (0 <= e <= 100):
                data = self.predict(a, b, c, d, e)

                if data == 0:
                    txt = "KHÔNG Ô NHIỄM"
                else:
                    txt = "Ô NHIỄM"

                self.lineEdit.setText(txt)
            else:
                QtWidgets.QMessageBox.warning(self, 'Thông báo', 'NHẬP SAI VUI LÒNG NHẬP LẠI!')



app = QtWidgets.QApplication(sys.argv)
window = PageMain()
window.show()
sys.exit(app.exec_())