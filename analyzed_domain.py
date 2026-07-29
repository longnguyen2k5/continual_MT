import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from datamodules.dataset import get_domain_data

load_dotenv()  # Load biến môi trường từ file .env nếu có

def main():
    print("🚀 BẮT ĐẦU TẢI DỮ LIỆU TỪ HUGGING FACE...")

    # 1. TASK 1: Y KHOA (VieMedEV) - "Trùm cuối" của từ vựng khó
    print("📥 1/3: Đang tải VieMedEV (Văn phong Y Khoa)...")
    # Lưu ý: Bộ này chỉ có tập test, và cột tiếng Anh tên là 'sentence2'
    dataset_med = get_domain_data(domain_name='medical', split_type='test', num_sample=500, cache_dir='./data')
    med_sentences = [ex["en"] for ex in dataset_med]

    # 2. TASK 2: CÔNG NGHỆ / PHẦN MỀM (KDE4)
    print("📥 2/3: Đang tải KDE4 (Văn phong IT)...")
    dataset_it = get_domain_data(domain_name='it', split_type='train', num_sample=500, cache_dir='./data')
    it_sentences = [ex["en"] for ex in dataset_it]

    # 3. TASK 3: PHIM ẢNH (OPUS-100 Subtitles)
    print("📥 3/3: Đang tải opus-100 (Văn phong Đời sống)...")
    dataset_sub = get_domain_data(domain_name='general', split_type='train', num_sample=500, cache_dir='./data')
    subtitles_sentences = [ex["en"] for ex in dataset_sub]

    # --- GỘP DỮ LIỆU ---
    all_sentences = med_sentences + it_sentences + subtitles_sentences
    labels = (
        ["Medical (Y khoa)"] * len(med_sentences) + 
        ["IT (Phần mềm)"] * len(it_sentences) + 
        ["Subtitles (Phim ảnh)"] * len(subtitles_sentences)
    )

    # --- CHẠY AI & VẼ BIỂU ĐỒ ---
    print("\n🧠 Đang chuyển đổi câu thành ma trận Vector (Embeddings)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_sentences, show_progress_bar=True)

    print("📉 Đang tính toán t-SNE để giảm chiều...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("🎨 Đang vẽ biểu đồ...")
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'Domain': labels
    })

    plt.figure(figsize=(10, 8))
    # Y khoa (Đỏ) - IT (Cam) - Phim ảnh (Xanh)
    colors = {'Medical (Y khoa)': '#d62728', 'IT (Phần mềm)': '#ff7f0e', 'Subtitles (Phim ảnh)': '#2ca02c'}
    
    for domain in colors.keys():
        subset = df[df['Domain'] == domain]
        plt.scatter(subset['x'], subset['y'], label=domain, color=colors[domain], alpha=0.7, edgecolors='w', s=50)

    plt.title("Phân tích Sự dịch chuyển Miền (Domain Shift) bằng t-SNE", fontsize=14, fontweight='bold')
    plt.xlabel("Chiều t-SNE 1")
    plt.ylabel("Chiều t-SNE 2")
    plt.legend(title="Miền dữ liệu (Domain)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("domain_shift_tsne.png", dpi=300, bbox_inches='tight')
    print("✅ Xong! Hãy mở file 'domain_shift_tsne.png' để chiêm ngưỡng thành quả.")


    # ==========================================
    # --- VẼ BIỂU ĐỒ HISTOGRAM CHIỀU DÀI CÂU ---
    # ==========================================
    print("\n📊 Đang vẽ biểu đồ phân phối chiều dài câu (Sequence Length)...")
    
    # 1. Tính số lượng từ của mỗi câu (cách đơn giản nhất là split bằng khoảng trắng)
    med_lengths = [len(s.split()) for s in med_sentences]
    it_lengths = [len(s.split()) for s in it_sentences]
    sub_lengths = [len(s.split()) for s in subtitles_sentences]

    # 2. Khởi tạo Figure mới
    plt.figure(figsize=(10, 6))
    
    # 3. Tìm chiều dài lớn nhất để chia bin (cột) cho đều
    max_len = max(max(med_lengths), max(it_lengths), max(sub_lengths))
    bins = np.linspace(0, max_len, 40) # Chia làm 40 cột

    # 4. Vẽ 3 histogram đè lên nhau (dùng alpha=0.5 để làm trong suốt)
    # Vẫn giữ nguyên tone màu đồng bộ với biểu đồ t-SNE
    plt.hist(med_lengths, bins=bins, alpha=0.6, label='Medical (Y khoa)', color='#d62728', edgecolor='white')
    plt.hist(it_lengths, bins=bins, alpha=0.6, label='IT (Phần mềm)', color='#ff7f0e', edgecolor='white')
    plt.hist(sub_lengths, bins=bins, alpha=0.6, label='Subtitles (Phim ảnh)', color='#2ca02c', edgecolor='white')

    # 5. Trang trí biểu đồ
    plt.title("Phân phối Chiều dài câu (Sequence Length) theo Miền dữ liệu", fontsize=14, fontweight='bold')
    plt.xlabel("Số lượng từ trong câu (Word count)", fontsize=12)
    plt.ylabel("Tần suất (Số lượng câu)", fontsize=12)
    plt.legend(title="Miền dữ liệu (Domain)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 6. Lưu thành file ảnh thứ 2
    plt.savefig("seq_len_histogram.png", dpi=300, bbox_inches='tight')
    print("✅ Xong! Hãy mở file 'seq_len_histogram.png' để xem phân phối chiều dài.")
    
if __name__ == "__main__":
    main()