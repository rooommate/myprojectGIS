import pandas as pd
# พิกัดของสวนสาธารณะในกรุงเทพมหานคร

PARK_COORDINATES = {
    'สวนสันติภาพ เขตราชเทวี': {
        'lat': 13.7540, 
        'lon': 100.5365, 
        'name_en': 'Santiphap Park',
        'district': 'Ratchathewi'
    },
    'สวนลุมพินี เขตปทุมวัน': {
        'lat': 13.7250, 
        'lon': 100.5383, 
        'name_en': 'Lumphini Park',
        'district': 'Pathumwan'
    },
    'อุทยานเบญจสิริ เขตคลองเตย': {
        'lat': 13.7363, 
        'lon': 100.5717, 
        'name_en': 'Benjakitti Park',
        'district': 'Khlong Toei'
    },
    'สวนเบญจกิติ เขตคลองเตย': {
        'lat': 13.7298, 
        'lon': 100.5564, 
        'name_en': 'Benjakiti Park (Old)',
        'district': 'Khlong Toei'
    },
    'สวนสาธารณะเฉลิมพระเกียรติ 6 รอบพระชนมพรรษา เขตบางคอแหลม': {
        'lat': 13.6989, 
        'lon': 100.5142, 
        'name_en': 'Chalerm Phra Kiat 6th Cycle Park',
        'district': 'Bang Kho Laem'
    },
    'สวนรมณีย์ทุ่งสีกัน เขตดอนเมือง': {
        'lat': 13.9124, 
        'lon': 100.6054, 
        'name_en': 'Romneeniphit Thungsikan Park',
        'district': 'Don Mueang'
    },
    'สวนกีฬารามอินทรา เขตบางเขน': {
        'lat': 13.8749, 
        'lon': 100.6052, 
        'name_en': 'Ram Inthra Sports Park',
        'district': 'Bang Khen'
    },
    'สวนวชิรเบญจทัศ เขตจตุจักร': {
        'lat': 13.8067, 
        'lon': 100.5502, 
        'name_en': 'Wachirabenchathat Park',
        'district': 'Chatuchak'
    },
    'สวนจตุจักร เขตจตุจักร': {
        'lat': 13.8024, 
        'lon': 100.5497, 
        'name_en': 'Chatuchak Park',
        'district': 'Chatuchak'
    },
    'สวนสมเด็จพระนางเจ้าสิริกิติ์ฯ เขตจตุจักร': {
        'lat': 13.8058, 
        'lon': 100.5530, 
        'name_en': 'Queen Sirikit Park',
        'district': 'Chatuchak'
    },
    'สวนหลวง ร.9 เขตประเวศ': {
        'lat': 13.7234, 
        'lon': 100.6436, 
        'name_en': 'King Rama IX Park',
        'district': 'Prawet'
    },
    'สวนเสรีไทย เขตบึงกุ่ม': {
        'lat': 13.7708, 
        'lon': 100.6467, 
        'name_en': 'Seri Thai Park',
        'district': 'Bueng Kum'
    },
    'สวนหนองจอก เขตหนองจอก': {
        'lat': 13.8765, 
        'lon': 100.8234, 
        'name_en': 'Nong Chok Park',
        'district': 'Nong Chok'
    },
    'สวน 60 พรรษา สมเด็จพระนางเจ้าฯ พระบรมราชินีนาถ เขตลาดกระบัง': {
        'lat': 13.7234, 
        'lon': 100.7456, 
        'name_en': '60th Anniversary Queen Park',
        'district': 'Lat Krabang'
    },
    'สวนพระนคร เขตลาดกระบัง': {
        'lat': 13.7198, 
        'lon': 100.7523, 
        'name_en': 'Phra Nakhon Park',
        'district': 'Lat Krabang'
    },
    'สวนหลวงพระราม 8 เขตบางพลัด': {
        'lat': 13.7876, 
        'lon': 100.4987, 
        'name_en': 'King Rama VIII Park',
        'district': 'Bang Phlat'
    },
    'สวนทวีวนารมย์ เขตทวีวัฒนา': {
        'lat': 13.7098, 
        'lon': 100.4789, 
        'name_en': 'Thawi Watthana Park',
        'district': 'Thawi Watthana'
    },
    'สวนเฉลิมพระเกียรติ 80 พรรษาพระบาทสมเด็จพระเจ้าอยู่หัว เขตบางกอกน้อย': {
        'lat': 13.7654, 
        'lon': 100.4876, 
        'name_en': 'Chalerm Phra Kiat 80th Birthday Park',
        'district': 'Bangkok Noi'
    },
    'สวนบางแคภิรมย์ เขตบางแค': {
        'lat': 13.6876, 
        'lon': 100.4234, 
        'name_en': 'Bang Kae Phirom Park',
        'district': 'Bang Khae'
    },
    'สวนธนบุรีรมย์ เขตทุ่งครุ': {
        'lat': 13.6543, 
        'lon': 100.4987, 
        'name_en': 'Thon Buri Rom Park',
        'district': 'Thung Khru'
    }
}

# ฟังก์ชันสำหรับดึงพิกัดของสวน
def get_park_coordinates(park_name):
    """
    ดึงพิกัดของสวนสาธารณะ
    Args:
        park_name (str): ชื่อสวนสาธารณะ
    Returns:
        dict: พิกัด lat, lon และข้อมูลเพิ่มเติม
    """
    return PARK_COORDINATES.get(park_name, None)

def get_all_parks_with_coordinates():
    """
    ดึงข้อมูลสวนทั้งหมดพร้อมพิกัด
    Returns:
        dict: ข้อมูลสวนทั้งหมด
    """
    return PARK_COORDINATES

# ฟังก์ชันสำหรับสร้างข้อมูลแผนที่
def create_map_data():
    """
    สร้างข้อมูลสำหรับแสดงในแผนที่
    Returns:
        list: รายการข้อมูลสำหรับแผนที่
    """
    map_data = []
    for park_name, coords in PARK_COORDINATES.items():
        map_data.append({
            'name': park_name,
            'name_en': coords['name_en'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'district': coords['district']
        })
    return pd.DataFrame(map_data) 