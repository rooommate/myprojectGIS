import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk


from park_coordinates import PARK_COORDINATES, create_map_data

st.set_page_config(
    page_title="Park PM2.5 in BKK Dashboard",
    # page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title(" PM2.5 Analysis Dashboard - สวนสาธารณะกรุงเทพฯ")
    st.markdown("แดชบอร์ดวิเคราะห์ค่า PM2.5 ของสวนสาธารณะต่างๆ ในกรุงเทพมหานคร")
    
    show_park_report()

def show_park_report():
    st.header("📋 รายงานการวิเคราะห์ PM2.5 สวนสาธารณะกรุงเทพฯ")
    st.markdown("รายงานค่า PM2.5 ของสวนสาธารณะต่างๆ ในกรุงเทพมหานคร พร้อมระบบกรองข้อมูลตามสถานที่และเวลา")
    
    # Load AllParkYear.csv
    try:
        df = pd.read_csv("Group_file/AllParkYear.csv")
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ AllParkYear.csv ในโฟลเดอร์ Group_file")
        return
    
    # Data preprocessing
    df_processed = preprocess_park_data(df)
    
    if df_processed.empty:
        st.error("ไม่สามารถประมวลผลข้อมูลได้")
        return
    
    # Filters in sidebar
    st.sidebar.header("Filter")
    
    # Location filter
    available_locations = ['ทั้งหมด'] + sorted(df_processed['สถานที่'].unique().tolist())
    selected_location = st.sidebar.selectbox(
        "เลือกสถานที่",
        available_locations
    )
    
    # Year filter
    available_years = ['ทั้งหมด'] + [str(year) for year in sorted(df_processed['ปี'].unique().tolist())]
    selected_year = st.sidebar.selectbox(
        "📅 เลือกปี",
        available_years
    )
    
    # Month filter
    months_thai = {
        'jan': 'มกราคม', 'feb': 'กุมภาพันธ์', 'mar': 'มีนาคม',
        'apr': 'เมษายน', 'may': 'พฤษภาคม', 'jun': 'มิถุนายน',
        'jul': 'กรกฎาคม', 'aug': 'สิงหาคม', 'sep': 'กันยายน',
        'oct': 'ตุลาคม', 'nov': 'พฤศจิกายน', 'dec': 'ธันวาคม'
    }
    
    available_months = ['ทั้งหมด'] + list(months_thai.values())
    selected_month = st.sidebar.selectbox(
        "📅 เลือกเดือน",
        available_months
    )
    
    filtered_df = filter_park_data(df_processed, selected_location, selected_year, selected_month)
    
    if filtered_df.empty:
        st.warning("⚠️ไม่พบข้อมูลตามเงื่อนไขที่เลือก")
        return
    
    show_park_metrics(filtered_df)
    
    show_park_visualizations(filtered_df, selected_location, selected_year, selected_month)
    
    show_park_data_table(filtered_df)

def preprocess_park_data(df):
    """Preprocess park data for analysis"""
    try:
        # Clean data
        df_clean = df.copy()
        
        # Convert to long format for easier analysis
        data_rows = []
        
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        months_thai = {
            'jan': 'มกราคม', 'feb': 'กุมภาพันธ์', 'mar': 'มีนาคม',
            'apr': 'เมษายน', 'may': 'พฤษภาคม', 'jun': 'มิถุนายน',
            'jul': 'กรกฎาคม', 'aug': 'สิงหาคม', 'sep': 'กันยายน',
            'oct': 'ตุลาคม', 'nov': 'พฤศจิกายน', 'dec': 'ธันวาคม'
        }
        
        for _, row in df_clean.iterrows():
            location = row['Dis_trict']
            year = int(row['ปี'])
            
            for month in months:
                try:
                    lowest = float(row[f'{month}_lowest_PM2.5']) if pd.notnull(row[f'{month}_lowest_PM2.5']) else 0
                    highest = float(row[f'{month}_highest_PM2.5']) if pd.notnull(row[f'{month}_highest_PM2.5']) else 0
                    average = float(row[f'{month}_average_PM2.5']) if pd.notnull(row[f'{month}_average_PM2.5']) else 0
                    exceeding = float(row[f'{month}_day_exceeding_month']) if pd.notnull(row[f'{month}_day_exceeding_month']) else 0
                    
                    data_rows.append({
                        'สถานที่': location,
                        'ปี': year,
                        'เดือน': months_thai[month],
                        'เดือนอังกฤษ': month,
                        'ค่าต่ำสุด': lowest,
                        'ค่าสูงสุด': highest,
                        'ค่าเฉลี่ย': average,
                        'จำนวนวันเกินมาตรฐาน': exceeding
                    })
                except (ValueError, TypeError):
                    # Skip problematic data
                    continue
        
        return pd.DataFrame(data_rows)
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def filter_park_data(df, location, year, month):
    """Filter park data based on user selections"""
    filtered = df.copy()
    
    if location != 'ทั้งหมด':
        filtered = filtered[filtered['สถานที่'] == location]
    
    if year != 'ทั้งหมด':
        year_int = int(year)
        filtered = filtered[filtered['ปี'] == year_int]
    
    if month != 'ทั้งหมด':
        filtered = filtered[filtered['เดือน'] == month]
    
    return filtered

def show_park_metrics(df):
    """Display summary metrics"""
    st.subheader("สรุปข้อมูลโดยรวม")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            avg_pm25 = pd.to_numeric(df['ค่าเฉลี่ย'], errors='coerce').mean()
            st.metric("PM2.5 เฉลี่ย", f"{avg_pm25:.1f} μg/m³" if not pd.isna(avg_pm25) else "N/A")
        
        with col2:
            max_pm25 = pd.to_numeric(df['ค่าสูงสุด'], errors='coerce').max()
            st.metric("ค่าสูงสุด", f"{max_pm25:.1f} μg/m³" if not pd.isna(max_pm25) else "N/A")
        
        with col3:
            total_exceeding = pd.to_numeric(df['จำนวนวันเกินมาตรฐาน'], errors='coerce').sum()
            st.metric("📅 วันเกินมาตรฐาน", f"{total_exceeding:.0f} วัน" if not pd.isna(total_exceeding) else "N/A")
        
        with col4:
            locations_count = df['สถานที่'].nunique()
            st.metric("จำนวนสถานที่", f"{locations_count} แห่ง")
            
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการแสดงสถิติ: {str(e)}")

def show_park_visualizations(df, location, year, month):
    """Display park visualizations"""
    st.subheader("กราฟการวิเคราะห์")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
        "ค่าเฉลี่ย PM2.5", 
        "แนวโน้มรายเดือน", 
        "เปรียบเทียบสถานที่", 
        "Box Plot วิเคราะห์", 
        "วันเกินมาตรฐาน",
        "แผนที่"
    ])
    
    with tab1:
        try:
            # Average PM2.5 by location
            if location == 'ทั้งหมด':
                df_numeric = df.copy()
                df_numeric['ค่าเฉลี่ย'] = pd.to_numeric(df_numeric['ค่าเฉลี่ย'], errors='coerce')
                
                avg_by_location = df_numeric.groupby('สถานที่')['ค่าเฉลี่ย'].mean().reset_index()
                avg_by_location = avg_by_location.dropna()
                avg_by_location = avg_by_location.sort_values('ค่าเฉลี่ย', ascending=True)
                
                if not avg_by_location.empty:
                    fig = px.bar(avg_by_location, 
                                x='ค่าเฉลี่ย', 
                                y='สถานที่',
                                title='ค่าเฉลี่ย PM2.5 แยกตามสถานที่',
                                orientation='h',
                                color='ค่าเฉลี่ย',
                                color_continuous_scale='RdYlGn_r')
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ไม่มีข้อมูลที่สามารถแสดงกราฟได้")
            else:
                # Show monthly trend for selected location
                df_numeric = df.copy()
                df_numeric['ค่าเฉลี่ย'] = pd.to_numeric(df_numeric['ค่าเฉลี่ย'], errors='coerce')
                
                monthly_data = df_numeric.groupby('เดือน')['ค่าเฉลี่ย'].mean().reset_index()
                monthly_data = monthly_data.dropna()
                
                if not monthly_data.empty:
                    month_order = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                                  'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
                    monthly_data['เดือน'] = pd.Categorical(monthly_data['เดือน'], categories=month_order, ordered=True)
                    monthly_data = monthly_data.sort_values('เดือน')
                    
                    fig = px.line(monthly_data, 
                                 x='เดือน', 
                                 y='ค่าเฉลี่ย',
                                 title=f'แนวโน้ม PM2.5 รายเดือน - {location}',
                                 markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ไม่มีข้อมูลที่สามารถแสดงกราฟได้")
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการสร้างกราฟ: {str(e)}")
    
    with tab2:
        # Monthly trend comparison
        try:
            if len(df['ปี'].unique()) > 1:
                df_numeric = df.copy()
                df_numeric['ค่าเฉลี่ย'] = pd.to_numeric(df_numeric['ค่าเฉลี่ย'], errors='coerce')
                
                monthly_trend = df_numeric.groupby(['เดือน', 'ปี'])['ค่าเฉลี่ย'].mean().reset_index()
                monthly_trend = monthly_trend.dropna()
                
                month_order = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                              'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
                monthly_trend['เดือน'] = pd.Categorical(monthly_trend['เดือน'], categories=month_order, ordered=True)
                monthly_trend = monthly_trend.sort_values('เดือน')
                
                fig = px.line(monthly_trend, 
                             x='เดือน', 
                             y='ค่าเฉลี่ย',
                             color='ปี',
                             title='แนวโน้ม PM2.5 รายเดือนแยกตามปี',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ต้องมีข้อมูลมากกว่า 1 ปีเพื่อแสดงแนวโน้ม")
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการสร้างกราฟแนวโน้ม: {str(e)}")
    
    with tab3:
        # Location comparison scatter plot
        try:
            if len(df['สถานที่'].unique()) > 1:
                location_data = df.groupby('สถานที่').agg({
                    'ค่าเฉลี่ย': 'mean',
                    'ค่าสูงสุด': 'max',
                    'ค่าต่ำสุด': 'min',
                    'จำนวนวันเกินมาตรฐาน': 'sum'
                }).reset_index()
                
                # Convert to numeric
                for col in ['ค่าเฉลี่ย', 'ค่าสูงสุด', 'ค่าต่ำสุด', 'จำนวนวันเกินมาตรฐาน']:
                    location_data[col] = pd.to_numeric(location_data[col], errors='coerce')
                
                location_data = location_data.dropna()
                
                fig = px.scatter(location_data,
                               x='ค่าเฉลี่ย',
                               y='จำนวนวันเกินมาตรฐาน',
                               size='ค่าสูงสุด',
                               hover_name='สถานที่',
                               title='เปรียบเทียบสถานที่: PM2.5 เฉลี่ย vs วันเกินมาตรฐาน',
                               color='ค่าเฉลี่ย',
                               color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ต้องเลือก 'ทั้งหมด' ในสถานที่เพื่อเปรียบเทียบ")
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการสร้างกราฟเปรียบเทียบ: {str(e)}")
    
    with tab4:
        # Box Plot Analysis - NEW FEATURE
        try:
            st.subheader("Box Plot Analysis - การวิเคราะห์การกระจายของข้อมูล")
            
            # Create different box plot options
            box_option = st.selectbox(
                "เลือกประเภท Box Plot:",
                ["PM2.5 แยกตามสถานที่", "PM2.5 แยกตามเดือน", "PM2.5 แยกตามปี"]
            )
            
            df_numeric = df.copy()
            for col in ['ค่าเฉลี่ย', 'ค่าสูงสุด', 'ค่าต่ำสุด']:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            df_numeric = df_numeric.dropna(subset=['ค่าเฉลี่ย'])
            
            if box_option == "PM2.5 แยกตามสถานที่" and len(df_numeric['สถานที่'].unique()) > 1:
                fig = px.box(df_numeric, 
                            x='สถานที่', 
                            y='ค่าเฉลี่ย',
                            title='Box Plot: การกระจายค่า PM2.5 แยกตามสถานที่',
                            color='สถานที่')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical interpretation
                st.info("""
                **การตีความ Box Plot:**
                - กล่อง (Box): แสดงความแตกต่างระหว่างควอไทล์ที่ 1 และ 3 (50% ของข้อมูล)
                - เส้นกลาง: ค่ามัธยฐาน (Median)
                - หนวด (Whiskers): ช่วงข้อมูลปกติ
                - จุดพิเศษ: ค่าผิดปกติ (Outliers)
                """)
                
            elif box_option == "PM2.5 แยกตามเดือน":
                month_order = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                              'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
                df_numeric['เดือน'] = pd.Categorical(df_numeric['เดือน'], categories=month_order, ordered=True)
                
                fig = px.box(df_numeric, 
                            x='เดือน', 
                            y='ค่าเฉลี่ย',
                            title='Box Plot: การกระจายค่า PM2.5 แยกตามเดือน',
                            color='เดือน')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
            elif box_option == "PM2.5 แยกตามปี" and len(df_numeric['ปี'].unique()) > 1:
                fig = px.box(df_numeric, 
                            x='ปี', 
                            y='ค่าเฉลี่ย',
                            title='Box Plot: การกระจายค่า PM2.5 แยกตามปี',
                            color='ปี')
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("ไม่สามารถสร้าง Box Plot ได้ เนื่องจากข้อมูลไม่เพียงพอ")
                
            # Additional box plot showing all three metrics
            st.subheader("เปรียบเทียบค่า PM2.5 (ต่ำสุด, เฉลี่ย, สูงสุด)")
            
            # Reshape data for multiple metrics comparison
            metrics_data = []
            for _, row in df_numeric.iterrows():
                metrics_data.extend([
                    {'สถานที่': row['สถานที่'], 'ค่า PM2.5': row['ค่าต่ำสุด'], 'ประเภท': 'ค่าต่ำสุด'},
                    {'สถานที่': row['สถานที่'], 'ค่า PM2.5': row['ค่าเฉลี่ย'], 'ประเภท': 'ค่าเฉลี่ย'},
                    {'สถานที่': row['สถานที่'], 'ค่า PM2.5': row['ค่าสูงสุด'], 'ประเภท': 'ค่าสูงสุด'}
                ])
            
            metrics_df = pd.DataFrame(metrics_data)
            
            if not metrics_df.empty:
                fig = px.box(metrics_df, 
                            x='ประเภท', 
                            y='ค่า PM2.5',
                            title='Box Plot: เปรียบเทียบค่า PM2.5 ประเภทต่างๆ',
                            color='ประเภท')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการสร้าง Box Plot: {str(e)}")
    
    with tab5:
        # Exceeding days analysis
        try:
            df_numeric = df.copy()
            df_numeric['จำนวนวันเกินมาตรฐาน'] = pd.to_numeric(df_numeric['จำนวนวันเกินมาตรฐาน'], errors='coerce')
            
            exceeding_data = df_numeric.groupby(['เดือน'])['จำนวนวันเกินมาตรฐาน'].sum().reset_index()
            exceeding_data = exceeding_data.dropna()
            
            month_order = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                          'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
            exceeding_data['เดือน'] = pd.Categorical(exceeding_data['เดือน'], categories=month_order, ordered=True)
            exceeding_data = exceeding_data.sort_values('เดือน')
            
            fig = px.bar(exceeding_data,
                        x='เดือน',
                        y='จำนวนวันเกินมาตรฐาน',
                        title='จำนวนวันที่ค่า PM2.5 เกินมาตรฐานแยกตามเดือน',
                        color='จำนวนวันเกินมาตรฐาน',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการสร้างกราฟวันเกินมาตรฐาน: {str(e)}")
    with tab6:
            st.header("🗺️ แผนที่สวนสาธารณะในกรุงเทพฯ พร้อมระดับความเสี่ยง")

    # สร้างข้อมูลแผนที่
            df_map = create_map_data()

    # คำนวณค่า PM2.5 เฉลี่ยของแต่ละสวน
            avg_pm = df.groupby("สถานที่")["ค่าเฉลี่ย"].mean().reset_index()
            avg_pm.rename(columns={"ค่าเฉลี่ย": "pm25_avg"}, inplace=True)

    # รวมเข้ากับพิกัด
            df_map = df_map.merge(avg_pm, left_on="name", right_on="สถานที่", how="left")

    # จัดระดับ
      # จัดระดับความเสี่ยง
    def classify_risk(val):
        if pd.isna(val):
            return "ไม่มีข้อมูล"
        elif val <= 25:
            return "🟢 ดี"
        elif val <= 50:
            return "🟡 ปานกลาง"
        else:
            return "🔴 เสี่ยงสูง"

    df_map["ระดับความเสี่ยง"] = df_map["pm25_avg"].apply(classify_risk)

    # แสดงผลเป็นตารางสรุป
    st.dataframe(df_map[["name", "pm25_avg", "ระดับความเสี่ยง"]])

    # ใช้ pydeck ในการแสดงผลแผนที่พร้อมสี

    color_map = {
        "🟢 ดี": [0, 200, 0],
        "🟡 ปานกลาง": [255, 215, 0],
        "🔴 เสี่ยงสูง": [255, 0, 0],
        "ไม่มีข้อมูล": [200, 200, 200],
    }

    df_map["color"] = df_map["ระดับความเสี่ยง"].apply(lambda x: color_map[x])

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=200,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=df_map["lat"].mean(),
        longitude=df_map["lon"].mean(),
        zoom=11
    )

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nค่าเฉลี่ย PM2.5: {pm25_avg:.1f}\n{ระดับความเสี่ยง}"})
    st.pydeck_chart(r)

def show_park_data_table(df):
    """Display detailed data table"""
    st.subheader("ตารางข้อมูลรายละเอียด")
    
    # Format the dataframe for display
    display_df = df.copy()
    display_df = display_df.round(1)
    
    # Add color coding for PM2.5 levels
    def color_pm25(val):
        if pd.isna(val):
            return ''
        elif val > 50:
            return 'background-color: #ffcccc'  # Light red for high levels
        elif val > 25:
            return 'background-color: #fff2cc'  # Light yellow for moderate levels
        else:
            return 'background-color: #ccffcc'  # Light green for good levels
    
    # Apply styling
    styled_df = display_df.style.applymap(color_pm25, subset=['ค่าเฉลี่ย', 'ค่าสูงสุด'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add download button
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label=" ดาวน์โหลดข้อมูล CSV",
        data=csv,
        file_name=f"PM25_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Add explanation
    st.info("""
    **คำอธิบายสีในตาราง:**
    - 🟢 เขียว: ค่า PM2.5 ≤ 25 μg/m³ (ดี)
    - 🟡 เหลือง: ค่า PM2.5 26-50 μg/m³ (ปานกลาง)  
    - 🔴 แดง: ค่า PM2.5 > 50 μg/m³ (สูง)
    
    **มาตรฐานคุณภาพอากาศ PM2.5:**
    - WHO แนะนำ: ไม่เกิน 15 μg/m³ ต่อปี
    - มาตรฐานไทย: ไม่เกิน 25 μg/m³ ต่อปี
    """)

if __name__ == "__main__":
    main()