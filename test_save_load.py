import os
import torch

def run_sanity_check_save_load(model_wrapper, test_method_name="mole"):
    """
    Script kiểm tra tự động xem Save/Load có hoạt động 100% chính xác không.
    """
    print(f"\n🧪 --- BẮT ĐẦU SANITY CHECK CHO: {test_method_name.upper()} ---")
    save_dir = "./test_checkpoints"
    task_name = "test_run"
    ckpt_path = os.path.join(save_dir, f"checkpoint_{task_name}.pt")
    
    # -------------------------------------------------------------
    # BƯỚC 1: TẠO GIẢ LẬP MODEL ĐÃ TRAIN QUA 2 TASK (MODEL 1)
    # -------------------------------------------------------------
    # Model 1 này đại diện cho mô hình vừa train xong
    model_src = model_wrapper # Mô hình Lightning của bạn
    model_src.eval()
    
    # Giả lập cho model_src trải qua 2 task để sinh ra 2 Task Experts / History
    for _ in range(2):
        for m in model_src.modules():
            if hasattr(m, 'on_task_start'):
                m.on_task_start()
                
    # Gán ngẫu nhiên một vài giá trị khác 0 cho trọng số để test cho chuẩn
    with torch.no_grad():
        for name, param in model_src.named_parameters():
            if "task_experts" in name or "history" in name or "task_keys" in name:
                param.add_(torch.randn_like(param) * 0.1)

    # -------------------------------------------------------------
    # BƯỚC 2: CHẠY DỰ ĐOÁN THỬ TRƯỚC KHI SAVE
    # -------------------------------------------------------------
    # Tạo một câu test cố định
    test_sentence = "Hello, this is a test sentence for checking load and save logic."
    
    print("🔄 Đang chạy suy luận trên Model gốc (Trước khi Save)...")
    with torch.no_grad():
        output_before_save = model_src.translate_sentence(test_sentence)
    print(f"📝 Kết quả dịch trước khi Save: '{output_before_save}'")

    # -------------------------------------------------------------
    # BƯỚC 3: THỰC HIỆN SAVE
    # -------------------------------------------------------------
    print("💾 Đang tiến hành Save Checkpoint...")
    model_src.save_smart_checkpoint(save_dir, task_name)
    
    # -------------------------------------------------------------
    # BƯỚC 4: TẠO MỘT MODEL TRỐNG TRƠN HOÀN TOÀN MỚI (MODEL 2)
    # -------------------------------------------------------------
    print("🏗️ Đang khởi tạo Model mới tinh (Chưa từng train, chưa có Task experts)...")
    # Khởi tạo mô hình mới từ đầu (chưa gọi on_task_start lần nào)
    # Giả sử bạn có hàm build_model_fn() để tạo lại instance
    # Hoặc bạn re-instantiate LightningModule
    model_dst = type(model_src)(model_src.cfg, model_src.tokenizer) # Tạo instance mới cùng class
    model_dst.to(model_src.device)
    model_dst.eval()

    # -------------------------------------------------------------
    # BƯỚC 5: LOAD CHECKPOINT VÀO MODEL MỚI
    # -------------------------------------------------------------
    print("📥 Đang nạp Checkpoint vào Model mới...")
    model_dst.load_smart_checkpoint(ckpt_path)

    # -------------------------------------------------------------
    # BƯỚC 6: CHẠY DỰ ĐOÁN TRÊN MODEL MỚI VÀ SO SÁNH
    # -------------------------------------------------------------
    print("🔄 Đang chạy suy luận trên Model mới (Sau khi Load)...")
    with torch.no_grad():
        output_after_load = model_dst.translate_sentence(test_sentence)
    print(f"📝 Kết quả dịch sau khi Load:  '{output_after_load}'")

    # -------------------------------------------------------------
    # BƯỚC 7: ĐÁNH GIÁ KẾT QUẢ (PASSED / FAILED)
    # -------------------------------------------------------------
    print("\n🔍 --- KẾT QUẢ KIỂM TRA ---")
    
    # Check 1: Câu dịch có giống nhau tuyệt đối không?
    if output_before_save == output_after_load:
        print("✅ PASSED 1/2: Câu dịch trước và sau khi Load TRÙNG KHỚP HOÀN TOÀN!")
    else:
        print("❌ FAILED 1/2: Câu dịch bị khác nhau! Có thể weights không được nạp đúng.")

    # Check 2: So sánh trực tiếp giá trị của 1 tensor bất kỳ
    # Tìm 1 tensor task_keys hoặc history để so sánh độ chênh lệch ma trận
    src_keys = [v for k, v in model_src.state_dict().items() if "task_keys" in k or "history_A" in k]
    dst_keys = [v for k, v in model_dst.state_dict().items() if "task_keys" in k or "history_A" in k]
    
    if len(src_keys) > 0 and len(dst_keys) > 0:
        max_diff = torch.max(torch.abs(src_keys[0].cpu() - dst_keys[0].cpu())).item()
        print(f"📊 Độ chênh lệch trọng số lớn nhất giữa 2 Model: {max_diff}")
        if max_diff < 1e-6:
            print("✅ PASSED 2/2: Ma trận trọng số trùng khớp tuyệt đối (Diff < 1e-6)!")
        else:
            print("❌ FAILED 2/2: Ma trận trọng số bị chênh lệch!")

    # Dọn dẹp file test
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("🧹 Đã dọn dẹp file test tạm.")