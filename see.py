import os
import json
import hashlib
import base64

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import albumentations as A
import numpy as np
from PIL import Image
import gradio as gr
import warnings

warnings.filterwarnings('ignore')

# ====================== 基础配置（核心：和你给的代码完全一致的背景处理） =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
CLS_NAMES = [
    "黑素瘤", "色素痣", "基底细胞癌",
    "日光性角化病", "良性角化病", "皮肤纤维瘤",
    "血管病变", "普通痣"
]
USER_DATA_FILE = "user_data.json"

# 背景图配置（和你给的代码完全一致）
IMG_FILE_NAME = "beijing.jpg"
IMG_ABS_PATH = os.path.abspath(IMG_FILE_NAME)


def convert_img_to_base64(img_path):
    try:
        with open(img_path, 'rb') as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except:
        return ""


BG_BASE64 = convert_img_to_base64(IMG_ABS_PATH)

# ====================== 8类皮肤病变详细危害（保留你提供的完整内容） =======================
DISEASE_HARM = {
    "黑素瘤": """
### 黑素瘤（Melanoma）
黑素瘤是恶性程度最高、预后最差的皮肤恶性肿瘤，起源于皮肤黑素细胞，可发生于身体任何部位，多见于中老年人，也可发生在青少年及儿童身上，其核心危害集中在侵袭性和转移性，具体表现如下：
- 侵袭性强，进展迅速：早期黑素瘤可能仅表现为色素斑片、结节，但若未及时发现，短期内会突破皮肤表皮层，侵犯真皮层、皮下组织，破坏周围正常皮肤组织，导致皮肤破溃、出血、糜烂，且难以愈合。
- 转移风险极高，危及生命：这是黑素瘤最致命的危害。肿瘤细胞可通过淋巴系统、血液循环早期转移至区域淋巴结，进而扩散到肺、肝、脑、骨骼等全身重要脏器，一旦发生远处转移，治疗难度大幅增加，5年生存率显著下降，严重威胁患者生命。
- 易被忽视，延误治疗：早期黑素瘤症状与普通色素痣相似，容易被误认为“普通痣”而忽视，等到出现明显破溃、增大、颜色异常时，往往已发展至中晚期，错失最佳治疗时机。
- 诱因明确，需重点警惕：长期紫外线照射、家族黑素瘤病史、反复摩擦刺激（如手掌、脚底、腰带区的痣）、色素痣突然恶变等，均会增加患病及进展风险。

**核心提示**：黑素瘤属于必须尽早发现、尽早手术切除的恶性肿瘤，早期干预后生存率较高，一旦发现皮肤色素病变有“不对称、边缘不规则、颜色不均匀、直径大于6mm、短期内迅速变化”的特点，需立即就医。
    """,
    "色素痣": """
### 色素痣（Nevus）
色素痣是由皮肤黑素细胞聚集形成的良性皮肤病变，几乎每个人身上都有，可发生于任何年龄、任何部位，多数情况下无明显危害，但其潜在风险需警惕，具体如下：
- 绝大多数良性，无健康威胁：普通色素痣边界清晰、颜色均匀、大小稳定，长期无变化，不会对健康造成影响，也不会恶变，无需特殊处理。
- 极少数存在恶变风险：长期受到摩擦、挤压、紫外线暴晒，或反复搔抓、外伤刺激（如抠挖、碰撞）的色素痣，尤其是位于手掌、脚底、腰带区、颈部、腋窝等易摩擦部位的痣，可能发生恶变，转变为黑素瘤等恶性肿瘤，一旦恶变，危害极大。
- 影响外观，可能造成心理困扰：部分色素痣长在面部、颈部等暴露部位，且体积较大、颜色较深，可能会影响个人外观，对部分人群造成心理压力，可通过正规医疗方式去除（需注意：自行去除易导致感染、留疤，甚至延误恶变的发现）。
- 特殊类型需警惕：先天性色素痣（出生时即存在），尤其是面积较大（直径大于20cm）的先天性巨痣，恶变风险高于普通色素痣，需长期随访观察。

**核心提示**：色素痣无需过度担心，但需定期观察，若发现痣的大小、形状、颜色突然变化，或出现破溃、出血、瘙痒、疼痛、周围出现卫星状小痣等情况，需及时就医排查恶变可能。
    """,
    "基底细胞癌": """
### 基底细胞癌（Basal Cell Carcinoma, BCC）
基底细胞癌是**最常见的皮肤恶性肿瘤**，起源于皮肤基底细胞，多见于中老年人，好发于面部、头皮、颈部等长期暴露于紫外线的部位，其恶性程度低于黑素瘤，但危害仍不可忽视，具体表现如下：
- 局部侵蚀性强，破坏组织：基底细胞癌生长速度较慢，但具有很强的局部侵袭性，会逐渐破坏周围的皮肤组织、肌肉，甚至侵犯骨骼、软骨，导致皮肤破溃、糜烂、结痂，形成难以愈合的溃疡，严重时会造成局部组织缺损、毁容（尤其是面部病变）。
- 极少发生远处转移，但易复发：与黑素瘤不同，基底细胞癌很少发生淋巴或全身远处转移，但其局部复发率较高，若手术切除不彻底，或治疗不及时，会反复复发，进一步破坏周围组织。
- 诱因明确，易被忽视：长期紫外线照射是主要诱因，此外，皮肤慢性炎症、外伤后瘢痕、长期接触砷剂等化学物质，也会增加患病风险。早期基底细胞癌症状不典型，可能表现为淡红色丘疹、结节，或类似“老年斑”的斑片，容易被忽视。
- 治疗不及时会加重危害：若长期未治疗，病变会持续扩大，侵蚀范围不断增加，不仅增加治疗难度，还可能导致面部畸形、功能障碍（如眼睑、鼻部受累，影响视力、呼吸等）。

**核心提示**：基底细胞癌属于恶性肿瘤，虽转移风险低，但需尽早手术切除，早期治疗后治愈率极高，复发率低，长期暴露在阳光下的人群需定期检查皮肤。
    """,
    "日光性角化病": """
### 日光性角化病（Actinic Keratosis, AK）
日光性角化病又称光线性角化病，是**最常见的癌前皮肤病变**，主要由长期紫外线照射导致皮肤角质形成细胞异常增生引起，多见于中老年人，好发于面部、头皮、手背、前臂等暴露部位，其核心危害在于恶变风险，具体如下：
- 明确的癌前病变，恶变风险高：日光性角化病本身不属于恶性肿瘤，但如果长期不干预、持续受到紫外线刺激，约5%-10%的病变会逐渐发展为鳞状细胞癌（一种恶性皮肤肿瘤），一旦恶变，会增加治疗难度和健康风险。
- 病变易复发，需长期监控：即使经过治疗（如冷冻、激光、外用药物），日光性角化病也可能复发，尤其是长期暴露在阳光下、皮肤屏障受损的人群，复发率更高，需定期随访观察。
- 局部症状影响生活质量：部分患者会出现病变部位瘙痒、刺痛、烧灼感，或皮肤干燥、粗糙、脱屑，严重时会出现轻微破溃、出血，影响日常生活。
- 提示皮肤损伤严重：出现日光性角化病，说明皮肤已受到长期、累积性的紫外线损伤，不仅易诱发该病，还会增加患其他皮肤癌（如基底细胞癌、黑素瘤）的风险。

**核心提示**：日光性角化病是“可预防、可治疗”的癌前病变，早期治疗可完全治愈，避免恶变。长期户外工作、经常暴晒的人群，需做好防晒措施，定期检查皮肤，发现异常及时就医。
    """,
    "良性角化病": """
### 良性角化病（Seborrheic Keratosis）
良性角化病又称脂溢性角化病，是一种常见的良性皮肤病变，起源于皮肤角质形成细胞，多见于中老年人（40岁以上人群多见），好发于面部、头皮、胸部、背部等部位，**无恶变风险，对健康基本无危害**，具体说明如下：
- 完全良性，无恶变可能：良性角化病是皮肤老化过程中出现的正常良性病变，与紫外线照射、皮肤老化、遗传等因素有关，其细胞形态正常，不会发生恶变，也不会转移，对身体健康无任何威胁。
- 主要危害是影响外观：病变表现为淡褐色、深褐色的斑片或丘疹，表面可能粗糙、油腻，随着年龄增长，数量可能增多、体积可能增大，长在暴露部位时，可能会影响个人外观，对部分注重美观的人群造成心理困扰。
- 极少数会出现轻微不适：多数患者无任何症状，少数人可能因病变部位受到摩擦、搔抓，出现轻微瘙痒、异物感，一般不影响生活质量。
- 无需治疗，必要时可美观性去除：由于无健康风险，良性角化病无需特殊治疗。若因影响外观想要去除，可通过冷冻、激光、电灼等正规医疗方式，避免自行抠挖、涂抹不明药物，以免导致感染、留疤。

**核心提示**：良性角化病与日光性角化病（癌前病变）、基底细胞癌（恶性肿瘤）外观可能相似，需注意区分，避免误判；若发现病变短期内迅速增大、颜色异常、破溃出血，需就医排查其他病变。
    """,
    "皮肤纤维瘤": """
### 皮肤纤维瘤（Dermatofibroma）
皮肤纤维瘤是一种常见的良性皮肤肿瘤，起源于皮肤真皮层的成纤维细胞，可发生于任何年龄，多见于中青年，好发于四肢伸侧、躯干等部位，**良性程度高，无恶变风险，对健康危害极小**，具体如下：
- 完全良性，无恶变、转移可能：皮肤纤维瘤是一种良性结节，细胞形态正常，生长缓慢，不会发生恶变，也不会通过淋巴、血液转移，对身体健康无威胁，无需过度担心。
- 局部轻微不适，多不影响生活：多数患者无任何症状，仅在触摸时能感觉到皮下结节（质地较硬、边界清晰、活动度较好）；少数人可能因结节受到摩擦、挤压，出现轻微瘙痒、压痛，一般不影响日常生活和工作。
- 可能与外伤、炎症有关：部分皮肤纤维瘤的发生与皮肤外伤、慢性炎症刺激有关（如蚊虫叮咬、皮肤划伤后愈合不当），但不会因这些诱因发生恶变。
- 无需治疗，必要时可切除：由于无健康风险，皮肤纤维瘤无需治疗。若结节体积较大、影响外观，或频繁出现瘙痒、疼痛，可通过手术切除，切除后一般不会复发，也不会留明显疤痕。

**核心提示**：皮肤纤维瘤需与黑素瘤、鳞状细胞癌等恶性病变区分，恶性病变多边界模糊、质地不均、生长迅速，而皮肤纤维瘤边界清晰、质地坚硬、长期稳定，若无法区分，可就医检查确认。
    """,
    "血管病变": """
### 血管病变（血管瘤、血管畸形等）
皮肤血管病变是一类以皮肤血管异常增生、扩张、畸形为主要表现的病变，涵盖范围较广，包括血管瘤、血管畸形、过敏性紫癜等，**多数为良性，但部分类型存在一定危害**，具体分类说明如下：
- 良性血管瘤（最常见）：多见于婴幼儿（出生后1-3个月出现），好发于头面部、颈部，表现为红色斑片、丘疹或结节，多数会在5-7岁自行消退，无明显危害。少数体积较大、生长迅速的血管瘤，可能会压迫周围正常组织（如眼部、鼻部血管瘤压迫器官，影响功能），或发生破溃、出血、感染，留下疤痕；长在暴露部位的血管瘤，也可能影响外观，对儿童心理造成影响。
- 血管畸形：属于先天性病变，出生时即存在，不会自行消退，表现为皮肤紫红色、青紫色斑片或结节，质地柔软。危害主要取决于病变部位和大小：面部血管畸形影响外观；四肢血管畸形可能导致肢体肿胀、畸形，影响活动；若病变累及重要器官（如咽喉、气道），可能会影响呼吸、吞咽等功能；部分血管畸形易破溃、出血，且出血后难以止血，反复出血可能导致贫血、感染。
- 其他血管病变：如过敏性紫癜，多见于儿童及青少年，与感染、过敏等因素有关，表现为下肢、臀部紫红色瘀点、瘀斑，可伴有腹痛、关节痛、蛋白尿等症状，若累及肾脏，可能导致肾炎，严重时影响肾功能；老年性血管瘤（樱桃状血管瘤），多见于中老年人，好发于躯干、上肢，表现为红色小丘疹，无恶变风险，仅影响外观。

**核心提示**：婴幼儿血管瘤需定期观察，若生长迅速、影响功能或外观，需及时干预；血管畸形需尽早明确诊断，根据类型进行针对性治疗；出现不明原因的皮肤瘀点、瘀斑，需就医排查是否为系统性血管病变。
    """,
    "普通痣": """
### 普通痣（Common Nevus）
普通痣是最常见的良性皮肤病变，属于色素痣的一种，由皮肤黑素细胞良性聚集形成，可发生于任何年龄、任何部位，**性质稳定，无恶变风险，对健康无危害**，具体说明如下：
- 完全良性，无健康威胁：普通痣边界清晰、颜色均匀（多为黑色、褐色、棕色）、大小稳定（直径一般小于6mm），生长缓慢，长期无变化，不会恶变，也不会对身体健康造成任何影响，无需特殊处理。
- 与恶性病变的区别明显：普通痣对称、边缘规则、颜色一致，而黑素瘤等恶性病变多不对称、边缘不规则、颜色不均匀、直径较大，且短期内会迅速变化，两者易区分。
- 仅可能影响外观：普通痣若长在面部、颈部等暴露部位，且颜色较深、体积较大，可能会影响个人外观，对部分人群造成轻微心理困扰，可通过正规医疗方式去除，无需过度纠结。
- 无需过度干预，避免刺激：普通痣无需治疗，日常避免反复摩擦、搔抓、抠挖，避免长期紫外线暴晒即可，过度刺激反而可能导致皮肤损伤，引发炎症，无需担心其恶变风险。

**核心提示**：普通痣与色素痣（易摩擦部位）、黑素瘤的核心区别在于“长期稳定、无异常变化”，若发现普通痣突然变大、变色、破溃、出血，需及时就医，排查是否为其他病变（而非普通痣本身恶变）。
    """
}


# ====================== 用户认证核心函数 =======================
def init_user_data():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)


def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def register_user(username, password, confirm_pwd):
    if not username or not password:
        return "❌ 用户名/密码不能为空！", "", "", ""
    if password != confirm_pwd:
        return "❌ 两次密码输入不一致！", username, password, ""

    init_user_data()
    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
        users = json.load(f)

    if username in users:
        return "❌ 用户名已存在！", username, "", ""

    users[username] = hash_password(password)
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

    return "✅ 注册成功！请登录", "", "", ""


def login_user(username, password):
    if not username or not password:
        return "❌ 用户名/密码不能为空！"

    init_user_data()
    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
        users = json.load(f)

    if username not in users:
        return "❌ 用户名不存在！"
    if users[username] != hash_password(password):
        return "❌ 密码错误！"

    return True


# ====================== 模型加载与预测函数 =======================
weights = MobileNet_V3_Large_Weights.DEFAULT
infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])


def load_model():
    model = mobilenet_v3_large(weights=None)
    num_classes = len(CLS_NAMES)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"✅ 模型加载成功 | 设备：{DEVICE} | 类别数：{num_classes}")
    return model


MODEL = load_model()


def predict_skin_lesion(img):
    if img is None:
        return "❌ 请先上传皮肤病变图片！", "❌ 无预测置信度"
    try:
        img = img.convert("RGB")
        img_np = np.array(img)
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            return "❌ 图片必须是RGB三通道格式！", "❌ 无预测置信度"

        img_aug = infer_transform(image=img_np)["image"]
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL(img_tensor)
            pred_probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(pred_probs, dim=1).item()
            pred_conf = pred_probs[0][pred_idx].item()

        pred_name = CLS_NAMES[pred_idx]
        conf_percent = round(pred_conf * 100, 2)
        return f"预测类别：{pred_name}", f"预测置信度：{conf_percent}%"
    except Exception as e:
        return f"❌ 预测失败：{str(e)[:50]}", "❌ 无预测置信度"


# ====================== 科普页面函数 =======================
def show_disease_info(disease_name):
    return DISEASE_HARM[disease_name]


# ====================== 自定义CSS（100%套用你给的代码，仅新增内容容器的半透明显式） =======================
CUSTOM_CSS = f"""
/* 核心：和你给的代码完全一致的背景样式 */
html, body {{
    width: 100vw !important;
    height: 100vh !important;
    background-image: url('{BG_BASE64}') !important;
    background-size: cover !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-color: transparent !important;
}}

.gradio-container {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

* {{
    background: transparent !important;
}}

/* 仅新增：内容容器的半透明样式（保证文字可读，不影响背景显示） */
.auth-container, .select-container, .kepu-container, .predict-container {{
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 15px !important;
    padding: 30px !important;
    margin: 20px auto !important;
    max-width: 800px !important;
}}
.kepu-container {{
    max-width: 1000px !important;
}}
.gr-button {{
    border-radius: 10px !important;
    padding: 10px 20px !important;
    margin: 5px !important;
}}
.gr-textbox {{
    border-radius: 10px !important;
    padding: 12px !important;
    background-color: rgba(255, 255, 255, 0.9) !important;
    margin: 8px 0 !important;
}}
.gr-markdown {{
    line-height: 1.6 !important;
    font-size: 15px !important;
    color: #333 !important;
}}
.gr-image {{
    background-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 10px !important;
    padding: 10px !important;
}}
.gr-logo, .gr-footer {{
    display: none !important;
}}
"""


# ====================== 界面搭建 =======================
def main():
    with gr.Blocks(css=CUSTOM_CSS) as demo:  # 移除theme，避免默认样式干扰背景
        # 1. 登录/注册页面
        with gr.Column(visible=True, elem_classes="auth-container") as auth_page:
            gr.Markdown("""
            # 🔐 皮肤病变分类系统
            ### 请先完成登录/注册
            """)

            with gr.Row():
                show_login_btn = gr.Button("登录", variant="primary")
                show_register_btn = gr.Button("注册", variant="secondary")

            # 登录表单
            with gr.Column(visible=True) as login_form:
                login_username = gr.Textbox(label="用户名", placeholder="输入你的用户名")
                login_password = gr.Textbox(label="密码", placeholder="输入你的密码", type="password")
                login_msg = gr.Textbox(label="登录状态", value="等待登录...", interactive=False)
                login_submit_btn = gr.Button("提交登录", variant="primary", size="lg")

            # 注册表单
            with gr.Column(visible=False) as register_form:
                reg_username = gr.Textbox(label="用户名", placeholder="设置你的用户名")
                reg_password = gr.Textbox(label="密码", placeholder="设置你的密码", type="password")
                reg_confirm_pwd = gr.Textbox(label="确认密码", placeholder="再次输入密码", type="password")
                reg_msg = gr.Textbox(label="注册状态", value="等待注册...", interactive=False)
                reg_submit_btn = gr.Button("提交注册", variant="secondary", size="lg")

            # 切换登录/注册
            def switch_to_login():
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    gr.Button(variant="primary"),
                    gr.Button(variant="secondary")
                )

            def switch_to_register():
                return (
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    gr.Button(variant="secondary"),
                    gr.Button(variant="primary")
                )

            show_login_btn.click(
                fn=switch_to_login,
                outputs=[login_form, register_form, show_login_btn, show_register_btn]
            )
            show_register_btn.click(
                fn=switch_to_register,
                outputs=[login_form, register_form, show_login_btn, show_register_btn]
            )

        # 2. 功能选择页面
        with gr.Column(visible=False, elem_classes="select-container") as select_page:
            gr.Markdown("""
            # 🎯 功能选择
            ### 请选择你要使用的功能
            """)
            with gr.Row():
                to_kepu_btn = gr.Button("📖 皮肤病变科普（查看危害）", variant="primary", size="lg")
                to_predict_btn = gr.Button("🔍 皮肤病变预测", variant="secondary", size="lg")
            back_to_auth_btn = gr.Button("🔙 返回登录/注册页", variant="secondary")

        # 3. 科普页面
        with gr.Column(visible=False, elem_classes="kepu-container") as kepu_page:
            gr.Markdown("""
            # 📖 皮肤病变科普
            ### 点击下方按钮，查看对应病变的常见危害
            """)
            with gr.Row():
                btn1 = gr.Button("黑素瘤", variant="secondary")
                btn2 = gr.Button("色素痣", variant="secondary")
                btn3 = gr.Button("基底细胞癌", variant="secondary")
                btn4 = gr.Button("日光性角化病", variant="secondary")
            with gr.Row():
                btn5 = gr.Button("良性角化病", variant="secondary")
                btn6 = gr.Button("皮肤纤维瘤", variant="secondary")
                btn7 = gr.Button("血管病变", variant="secondary")
                btn8 = gr.Button("普通痣", variant="secondary")

            harm_info = gr.Markdown(label="病变危害详情", value="请点击上方按钮查看对应病变的危害信息...")
            back_to_select_btn1 = gr.Button("🔙 返回功能选择", variant="secondary")

        # 4. 预测页面
        with gr.Column(visible=False, elem_classes="predict-container") as predict_page:
            gr.Markdown("""
            # 🧑⚕️ ISIC皮肤病变分类系统
            ### 基于MobileNetV3Large模型 | 支持8类皮肤病变识别
            """)
            back_to_select_btn2 = gr.Button("🔙 返回功能选择", variant="secondary")

            with gr.Row():
                img_input = gr.Image(
                    type="pil",
                    label="上传皮肤病变图片",
                    height=400,
                    width=400,
                    image_mode="RGB"
                )
                with gr.Column():
                    pred_cls = gr.Textbox(label="📌 预测结果", value="等待预测...", interactive=False)
                    pred_conf = gr.Textbox(label="📊 预测置信度", value="等待预测...", interactive=False)

            predict_btn = gr.Button("开始预测", variant="primary", size="lg")
            predict_btn.click(predict_skin_lesion, inputs=img_input, outputs=[pred_cls, pred_conf])
            gr.Markdown("> 模型仅供学习研究使用，不构成医疗诊断建议")

        # ====================== 交互逻辑 ======================
        # 注册提交
        reg_submit_btn.click(
            register_user,
            inputs=[reg_username, reg_password, reg_confirm_pwd],
            outputs=[reg_msg, reg_username, reg_password, reg_confirm_pwd]
        )

        # 登录提交
        def handle_login(username, password):
            res = login_user(username, password)
            if res is True:
                return "✅ 登录成功！正在跳转...", "", "", gr.Column(visible=False), gr.Column(visible=True)
            else:
                return res, username, password, gr.Column(visible=True), gr.Column(visible=False)

        login_submit_btn.click(
            handle_login,
            inputs=[login_username, login_password],
            outputs=[login_msg, login_username, login_password, auth_page, select_page]
        )

        # 功能选择 → 科普/预测
        to_kepu_btn.click(lambda: (gr.Column(visible=False), gr.Column(visible=True)), outputs=[select_page, kepu_page])
        to_predict_btn.click(lambda: (gr.Column(visible=False), gr.Column(visible=True)),
                             outputs=[select_page, predict_page])

        # 科普按钮点击
        btn1.click(lambda: show_disease_info("黑素瘤"), outputs=harm_info)
        btn2.click(lambda: show_disease_info("色素痣"), outputs=harm_info)
        btn3.click(lambda: show_disease_info("基底细胞癌"), outputs=harm_info)
        btn4.click(lambda: show_disease_info("日光性角化病"), outputs=harm_info)
        btn5.click(lambda: show_disease_info("良性角化病"), outputs=harm_info)
        btn6.click(lambda: show_disease_info("皮肤纤维瘤"), outputs=harm_info)
        btn7.click(lambda: show_disease_info("血管病变"), outputs=harm_info)
        btn8.click(lambda: show_disease_info("普通痣"), outputs=harm_info)

        # 返回逻辑
        back_to_select_btn1.click(lambda: (gr.Column(visible=False), gr.Column(visible=True)),
                                  outputs=[kepu_page, select_page])
        back_to_select_btn2.click(lambda: (gr.Column(visible=False), gr.Column(visible=True)),
                                  outputs=[predict_page, select_page])
        back_to_auth_btn.click(lambda: (gr.Column(visible=False), gr.Column(visible=True)),
                               outputs=[select_page, auth_page])

    # 启动服务（和你给的代码完全一致）
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )


if __name__ == "__main__":
    main()