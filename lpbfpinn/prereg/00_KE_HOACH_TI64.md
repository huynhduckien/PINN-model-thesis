# Prereg: PINN nhiệt cho overhang không support, Ti-6Al-4V, LPBF

Trạng thái: **ĐÃ KHÓA (2026-10-10)** cho các giá trị ở mục 2. Chưa chạy huấn luyện nào.
Mọi thay đổi sau khi khóa phải ghi lại kèm ngày giờ, không sửa lặng lẽ.

## 1. Phạm vi

- Vật liệu: Ti-6Al-4V.
- 300 W, 0.8 m/s, r_b = 40 µm, lớp bột 60 µm, một đường quét.
- Hình học: overhang với mặt nghiêng 45° hướng xuống. Không mô phỏng vệt thường.
- Đầu ra: T(x,y,z,t) đầy đủ trên miền overhang. Chỉ báo là giá trị tuyệt đối tại vị trí cố định (mục 7).
- Không dự đoán ứng suất, biến dạng, hay võng (giai đoạn sau).
- Không hiệu chỉnh FEA trong suy luận (nhánh riêng sau này).
- Không dùng FD/FEA trong loss. FD chỉ là tham chiếu kiểm chứng.
- Bay hơi giữ (300 W, khôi phục bay hơi).
- DeepXDE làm thư viện nền. Code viết mới trong `lpbfpinn/`.
- Cấu hình bằng dataclass Python; mỗi lượt chạy có manifest JSON.

## 2. Giá trị vật liệu và quá trình (đã khóa)

| Tham số | Giá trị | Nguồn / ghi chú |
|---|---|---|
| T_sol | 1878 K | Cook & Ritchie 2023, bảng 2 |
| T_liq | 1928 K | Cook & Ritchie 2023, bảng 2 |
| ΔT = T_liq − T_sol | 50 K | Hệ quả |
| L_f | 2.90×10⁵ J/kg | Cook & Ritchie 2023, bảng 2 |
| T_b | 3315 K | Cook & Ritchie 2023, bảng 2 |
| L_v(0) | 9.83×10⁶ J/kg | Cook & Ritchie 2023, bảng 2 |
| L_v(T) | L_v(0)·√(1 − T²/T_c²), T_c = 7845 K | Cook & Ritchie 2023, phương trình 12 |
| M (khối lượng mol) | 0.0467 kg/mol | Cook & Ritchie 2023, bảng 2 (trung bình có trọng số hợp kim) |
| φ (hệ số bay hơi) | 0.82 | Cook & Ritchie 2023, phương trình 11 |
| h_c (đối lưu, biên trên) | 40 W/m²·K | Cook & Ritchie 2023, phương trình 14 |
| α_s (hấp thụ, chuẩn) | 0.27 | Cook & Ritchie 2023, tập chế độ dẫn nhiệt, 1.07 µm |
| φ_bột (độ rỗng) | 0.60 | Mật độ đóng gói khoảng 40% ([Springer 10.1007/s40516-022-00185-3](https://link.springer.com/article/10.1007/s40516-022-00185-3)) |
| d50 | Chưa có nguồn | Giá trị IN718 cũ (30 µm). Không dùng cho đến khi có nguồn. |
| k_gas (argon) | Chưa có nguồn | Giả định 0.018 W/m·K. Không dùng cho đến khi kiểm tra. |
| ε (bức xạ, biên trên) | Chưa có nguồn | Giả định 0.3, chỉ chạy độ nhạy. |
| f_L | 1 (chuẩn) | Độ nhạy 2.8 |

Các giá trị có nguồn cho k_s, c_p, ρ theo nhiệt độ: lấy từ Cook & Ritchie (bảng 2), nội suy tuyến tính giữa các điểm đã cho. Đây là phương pháp của bài này.

## 3. Hấp thụ laser

- Chuẩn: α_s = 0.27 (Cook & Ritchie). Bài này đo trong LPBF, 1.07 µm, bề mặt nóng chảy, chế độ dẫn nhiệt.
- Độ nhạy: α_s = 0.33 (giá trị khi tập toàn bộ thí nghiệm, Cook & Ritchie bảng 3).
- Lưu ý: giá trị 0.27 được hiệu chỉnh trong mô hình có phản xạ đa lần. Mô hình của chúng ta không có phản xạ đa lần. Đây là một khác biệt cần ghi trong luận văn.
- Tương quan Moser cho bột (α_eff, β, k_bột) tính từ α_s = 0.27: α_eff ≈ 0.355, 1/β ≈ 28.5 µm. Chưa tính lại k_bột với k_gas chưa có nguồn.

## 4. Bay hơi trên biên trên

Cook & Ritchie, phương trình 11:

```
q_evap(T) = φ · L_v(T) · √(M / (2πRT)) · p_amb · exp[ −(L_v(0)·M/R)·(1/T − 1/T_b) ]
L_v(T)    = L_v(0) · √(1 − T²/T_c²)
```

Lưu ý khác với bản trước: prefactor dùng L_v(T), số mũ dùng L_v(0); có thêm φ = 0.82.

Thông lượng so với đỉnh laser (α_s = 0.27, 3.22×10¹⁰ W/m²):

| T (K) | q_evap / q_đỉnh |
|---|---|
| 3000 | 0.22% |
| 3300 | 1.11% |
| 4000 | 17.9% |
| 5000 | 226% |

Quy tắc: báo cáo tỷ lệ bay hơi nếu nhiệt độ đỉnh vượt 4000 K. Nếu vượt 5000 K thì mô hình không hợp lệ (không có keyhole, không có áp suất giật).

## 5. Lớp chuyển pha và lịch sử pha

**Tỷ lệ lỏng tuyến tính** (Cook & Ritchie phương trình 5; quy tắc đòn bẩy của Moser):

```
g(T) = 0                 T < T_sol
g(T) = (T − T_sol)/ΔT    T_sol ≤ T ≤ T_liq
g(T) = 1                 T > T_liq
```

Nhiệt dung biểu kiến: c_eff(T) = c_p(T) + L_f·dg/dT, với dg/dT = 1/ΔT trong khoảng. Đỉnh biểu kiến ≈ 2.90×10⁵/50 = 5800 J/kg·K.

Hàm Gauss σ = 150 K của IN718 đã bỏ.

**Lịch sử pha M(x,t):** cực đại tích lũy theo thời gian của g(T), không giảm. Số vòng lặp ngoài là 4 (lựa chọn của code cũ, không có trong hai bài). Tiêu chí dừng: đọc lại `prereg/08` của nhánh cũ trước khi chạy và ghi lại vào đây.

## 6. Biên

- Đáy: T = T_amb (cứng).
- Hai biên bên: cách nhiệt, ∂T/∂n = 0. Cần kiểm tra trong code.
- Biên trên:

```
−k·∂T/∂z = q_evap(T) + h_c·(T − T_amb) + ε·σ_SB·(T⁴ − T_amb⁴)
```

Với h_c = 40 W/m²·K (Cook & Ritchie: không ảnh hưởng đáng kể trong vùng nóng). Chạy độ nhạy ε ∈ {0.2, 0.3, 0.6} và h_c ∈ {20, 40}.

## 7. Chỉ báo overhang

Giá trị tuyệt đối tại vị trí cố định (ví dụ điểm dưới đáy vùng treo):
- nhiệt độ đỉnh tại vị trí cố định;
- thời gian lưu trên T_sol;
- tốc độ nguội qua khoảng T_sol → T_liq.

Chưa khóa: vị trí cụ thể và công thức. Sẽ khóa trước khi chạy bài chính.

## 8. Lấy mẫu thích ứng và kiểm chứng

**Lấy mẫu thích ứng:** tăng mật độ điểm gần các đẳng nhiệt T_sol và T_liq, và gần đỉnh laser. Báo cáo tỷ lệ điểm trong dải chuyển pha.

**Stefan 1D (cổng bắt buộc trước bài chính):**
- Nghiệm giải tích (Neumann, Carslaw & Jaeger) chỉ có cho biên chảy sắc. Dải 50 K không có nghiệm đóng.
- Kiểm chứng: (a) so với nghiệm Neumann trong giới hạn ΔT → 0; (b) với ΔT = 50 K, kiểm tra hội tụ khi ΔT giảm dần và kiểm tra cân bằng năng lượng.
- Ngưỡng đạt: lấy từ `prereg/08`, khóa trước khi chạy.

**Các cổng khác:**
1. Cân bằng năng lượng trong mọi lượt chạy (nhiệt đưa vào = tích lũy + tổn thất).
2. Hội tụ theo số điểm lấy mẫu và số vòng lặp ngoài.
3. Độ nhạy: α_s (0.27, 0.33), f_L (1, 2.8), h_c, ε, φ, φ_bột.
4. Đối chiếu với FD không tham gia loss.

## 9. Việc còn mở (không chặn việc viết code)

1. d50 bột Ti-6Al-4V: cần nguồn.
2. k_gas argon: cần nguồn hoặc xác nhận.
3. ε: cần nguồn Ti, hoặc giữ như giả định và chạy độ nhạy.
4. Ngưỡng cổng và tiêu chí dừng vòng lặp ngoài: đọc `prereg/08`.
5. Chỉ báo overhang cụ thể (mục 7).

## 10. Nhật ký thay đổi

- 2026-10-10: khóa mục 2 theo quyết định của bạn. Thay α_s 0.35 → 0.27; T_sol/T_liq 1873/1923 → 1878/1928; L_v 9.0 MJ/kg hằng số → công thức Cook; M 0.04787 → 0.0467; φ_bay hơi 1 → 0.82; h_c 20 → 40.
