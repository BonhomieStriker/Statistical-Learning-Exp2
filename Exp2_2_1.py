from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
boston = load_boston()
# print(boston.DESCR)
# print("Max Price：", np.max(boston.target))   # 50
# print("Min Price：",np.min(boston.target))    # 5
# print("Mean Price：", np.mean(boston.target))   # 22.532806324110677
x = boston.data
y = boston.target

# Data Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

# Data normalization
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# SVR training with different kernels
# linear kernel
svr_lr = SVR(kernel="linear")
svr_lr.fit(x_train, y_train)
svr_lr_y_predict = svr_lr.predict(x_test)
# Polynomial kernel
svr_poly = SVR(kernel="poly")
svr_poly.fit(x_train, y_train)
svr_poly_y_predict = svr_poly.predict(x_test)
# RBF kernel
svr_rbf = SVR(kernel="rbf")
svr_rbf.fit(x_train, y_train)
svr_rbf_y_predict = svr_rbf.predict(x_test)

#Result Visualization
plt.plot(y_test)
plt.plot(svr_lr_y_predict)
plt.plot(svr_poly_y_predict)
plt.plot(svr_rbf_y_predict)
plt.legend(['True Price','Linear kernel','Polynomial kernel','RBF kernel'])
plt.title("House Price Prediction of SVR with different kernels")
plt.show()
error_lr = np.array(y_test).reshape(-1) - np.array(svr_lr_y_predict).reshape(-1);
error_poly = np.array(y_test).reshape(-1) - np.array(svr_poly_y_predict).reshape(-1);
error_rbf = np.array(y_test).reshape(-1) - np.array(svr_rbf_y_predict).reshape(-1);
plt.plot(error_lr)
plt.plot(error_poly)
plt.plot(error_rbf)
plt.legend(['Linear kernel','Polynomial kernel','RBF kernel'])
plt.title("Prediction error of SVR with different kernels")
plt.show()
# Model Evaluation
print("__________")
print("Linear Kernel")
print("Score：", svr_lr.score(x_test, y_test))
print("R2：", r2_score(y_test, svr_lr_y_predict))
print("MSE:", mean_squared_error(ss_y.inverse_transform(y_test),
                                 ss_y.inverse_transform(svr_lr_y_predict)))
print("MAE:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                  ss_y.inverse_transform(svr_lr_y_predict)))
print("__________")
print("Polynomial Kernel")
print("Score：", svr_poly.score(x_test, y_test))
print("R2：", r2_score(y_test, svr_poly_y_predict))
print("MSE:", mean_squared_error(ss_y.inverse_transform(y_test),
                                 ss_y.inverse_transform(svr_poly_y_predict)))
print("MAE:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                  ss_y.inverse_transform(svr_poly_y_predict)))
print("__________")
print("RBF Kernel")
print("Score：", svr_rbf.score(x_test, y_test))
print("R2：", r2_score(y_test, svr_rbf_y_predict))
print("MSE:", mean_squared_error(ss_y.inverse_transform(y_test),
                                 ss_y.inverse_transform(svr_rbf_y_predict)))
print("MAE:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                  ss_y.inverse_transform(svr_rbf_y_predict)))