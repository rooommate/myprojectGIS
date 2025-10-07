# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# def show_forecast_page():
#     st.header("📈 การพยากรณ์ค่า PM2.5 ล่วงหน้า 4 ปี")
#     st.markdown("ใช้ข้อมูลจาก AllParkYear.csv เพื่อทำนายแนวโน้มค่าเฉลี่ย PM2.5 ของสวนสาธารณะในอนาคต")

#     try:
#         df = pd.read_csv("Group_file/AllParkYear.csv")
#     except FileNotFoundError:
#         st.error("ไม่พบไฟล์ AllParkYear.csv ในโฟลเดอร์ Group_file")
#         return

#     df_processed = preprocess_park_data(df)
#     if df_processed.empty:
#         st.error("ไม่สามารถประมวลผลข้อมูลได้")
#         return

#     # ใช้ค่าเฉลี่ย PM2.5 ต่อปี (รวมทุกสวน)
#     df_yearly = df_processed.groupby("ปี")["ค่าเฉลี่ย"].mean().reset_index()
#     df_yearly["ค่าเฉลี่ย"] = pd.to_numeric(df_yearly["ค่าเฉลี่ย"], errors="coerce")

#     st.subheader("ข้อมูลค่าเฉลี่ย PM2.5 ต่อปี")
#     st.dataframe(df_yearly)

#     # แบ่งข้อมูล train/test
#     X = df_yearly[["ปี"]]
#     y = df_yearly["ค่าเฉลี่ย"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     models = {
#         "Linear Regression": LinearRegression(),
#         "Random Forest": RandomForestRegressor(),
#         "Gradient Boosting": GradientBoostingRegressor(),
#         "SVM (RBF)": SVR(kernel="rbf")
#     }

#     results = []
#     future_preds = {}
#     last_year = int(df_yearly["ปี"].max())
#     future_years = np.array([last_year + i for i in range(1, 5)]).reshape(-1, 1)
#     future_scaled = scaler.transform(future_years)

#     for name, model in models.items():
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)

#         r2 = r2_score(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#         results.append({
#             "Model": name,
#             "R² Score": r2,
#             "MAE": mae,
#             "RMSE": rmse
#         })

#         # ทำนาย 4 ปีข้างหน้า
#         future_preds[name] = model.predict(future_scaled)

#     st.subheader("ประสิทธิภาพโมเดล (Test 20%)")
#     results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
#     st.dataframe(results_df, use_container_width=True)

#     st.subheader("การพยากรณ์ค่า PM2.5 ล่วงหน้า 4 ปี")
#     forecast_df = pd.DataFrame({
#         "ปี": [last_year + i for i in range(1, 5)]
#     })
#     for name, preds in future_preds.items():
#         forecast_df[name] = preds

#     st.dataframe(forecast_df, use_container_width=True)

#     # แสดงกราฟเปรียบเทียบ
#     fig = px.line(forecast_df, x="ปี", y=forecast_df.columns[1:], 
#                   title="แนวโน้มการพยากรณ์ค่า PM2.5 4 ปีข้างหน้า", markers=True)
#     st.plotly_chart(fig, use_container_width=True)