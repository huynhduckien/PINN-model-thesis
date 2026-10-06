# Đăng ký trước 03 — M2′: `B3W0` với đầu ra cắt θ ≥ 0 (hướng A)

Ngày: 2026-10-06 15:18 (đồng hồ). Người dùng chọn hướng A rồi mới sang hướng 4 (hiệu chỉnh/bất định hấp thụ, sẽ là đăng ký 04). **Commit file này trước khi viết mã và trước khi chạy bất kỳ mốc nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất.

## 1. Căn cứ (số đo của chẩn đoán không huấn luyện, `runs/M2/diag_energy.json`; không phải mốc)

M2 (đăng ký 02, mục 14) **TRƯỢT và vẫn là TRƯỢT**; bản ghi đó không được sửa. Chẩn đoán trên 2^18 điểm đều (seed 777) cho thấy:
- E_res của M2 là −25.0 mJ, trong đó **−25.8 mJ nằm ở vùng T < T_amb** (28.5% số điểm); mọi vùng còn lại cộng lại +0.8 mJ. Với B3 gốc: tổng −7.8 mJ, vùng T < T_amb −8.0 mJ.
- Khi cắt θ tại 0 (không huấn luyện lại): E_res của M2 = **+0.03 mJ**, của B3 gốc = **+0.11 mJ**. Trong vùng T < T_amb, nhiệt độ âm là lỗi của mạng (nguyên lý cực đại: θ ≥ 0 khi nguồn ≥ 0, điều kiện đầu và biên đều bằng 0), và ở đó PDE cũng không thỏa (RMS residual 0.025).
- Vị trí điểm âm: 64% phía trước laser, 29% sát laser. Độ hụt T_min tăng dần theo Adam (0.94 → 1.23%) và không giảm ở L-BFGS. Vùng nóng (> 1000 K) của M2 có RMS residual 0.0635, so với 0.0398 ở B3 gốc.
- Hệ quả: gate (a) và (b) của M2 gần như chỉ đo lỗi nhiệt độ âm ở vùng lạnh. Gate E_res là tổng có dấu nên không phản ánh sai số cục bộ ở vùng nóng.
- Các checkpoint cũ trượt theo kiểu khác: chẩn đoán cũ của B1kc cho nguồn giả dương +47.8 mJ ở dải 298–1000 K, không phải ở vùng T < T_amb; phép cắt không được kỳ vọng làm chúng qua cổng (kiểm ở S2.10(e)).

## 2. Thay đổi mô hình (không đổi tiêu chí)

Bậc mới `B3W0c` trong `config.py`: cùng hình học nêm (y_a = 0), cùng vật lý B3 như `B3W0`, khởi động từ B3, **nhưng** đầu ra của mạng được cắt: θ = max(θ_mạng, 0), gắn bằng cách bọc `net._output_transform` sau `L.build_net` (không sửa `legacy/`). Vùng cắt có θ = 0 đúng, và PDE thỏa chính xác ở đó vì nguồn bằng 0; đạo hàm bậc hai của max bằng 0 gần như mọi nơi nên phần dư không thấy điểm gãy. Cắt không đổi giá trị ở nơi θ_mạng ≥ 0, nên bể chảy và chu trình nhiệt ở vùng nóng giữ nguyên; chỉ T_min và E_res bị ảnh hưởng. **Gate (b) nay qua theo cấu trúc (T_min = T_amb), nên không còn là bằng chứng chất lượng; gate (a), gate (c) và `check_physics` vẫn có nghĩa vì tính trên trường đã cắt.** Tag khác nhờ tên bậc `B3W0c`; `tag_of("B3W0")` phải giữ `B3W0_fff2a3d4`.

## 3. Mốc test tầng 2c (chạy trước M2′; trượt mốc nào thì dừng, báo cáo, không chạy M2′)

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S2.10 (a) | Cắt đúng: trọng số M2 (có θ < 0 khi không cắt) trên 2^16 điểm | không cắt: min θ < 0 (mốc không rỗng); có cắt: min θ ≥ 0; chênh lớn nhất so với không cắt ở nơi θ ≥ 0 bằng **0**; có cắt = 0 ở nơi θ < 0 |
| (b) | Phần dư và gradient: `L.pde` trên 2^14 điểm, float32 và float64, backward tới trọng số | mọi giá trị hữu hạn, gradient trọng số hữu hạn; ở float64 mạng giữ dtype float64 |
| (c) | **Canary:** B3 (hình học B3) có cắt, đo bằng `L.box`, `L.melt_pool`, `L.probe` | rộng/sâu/dài 166 ± 2 / 90 ± 2 / 516 ± 4 µm; (tích + bay hơi)/nạp 0.805 ± 0.01; |E_res| ≤ 10 mJ; T_min ≥ 297.999 K |
| (d) | Chỉ báo của B3 có cắt (`analysis.net_indicator`) so với `runs/M1/indicator.json` | mọi số đo kích thước, T_peak, T_pk, thời gian lưu, tốc độ nguội trùng trong 1e-4 tương đối (xác nhận M1 vẫn hợp lệ) |
| (e) | **Khả năng phân biệt:** cổng đầy đủ có cắt trên `B1k_4d5bf24c`, `B1kc_e7de526d`, `B1f1.4_bd84c1f5` | cả ba vẫn **TRƯỢT** ít nhất một gate (a hoặc c); B3 vẫn PASS |
| (f) | Hồi quy và rò rỉ: `tag_of("B3W0")` = `B3W0_fff2a3d4`, `tag_of("B3W0c")` khác nó; hash `legacy/ckpt` và `lpbf_pinn.py` không đổi | đúng |
| (g) | Khói đầu-cuối `runner.py run B3W0c` ngân sách tí hon (Adam 20, khối 5 vòng, tối đa 12), thư mục tạm | ra `run.json` DONE, `gate_full` có, T_min ≥ 297.999 K, loss hữu hạn, tag ≠ tag `B3W0` |

Trần GPU tầng 2c: 10 phút (dừng ở 15). Dự đoán: đạt cả bảy ngay lượt đầu 65%; rủi ro chính là (e), nếu một checkpoint cũ qua cổng sau khi cắt thì cổng mất khả năng phân biệt và phải dừng thảo luận.

## 4. M2′

Cùng quy trình và ngân sách M2: bậc `B3W0c`, khởi động từ `B3_84d0289e` (chỉ đọc), 4 seed Sobol cố định (K = 0), seed 42; Adam 3,000 bước lr 1e-4, cổng nhẹ mỗi 500 bước (chỉ ghi); L-BFGS float64 khối 500, tối đa 2,500 vòng, cổng đầy đủ sau mỗi khối, dừng khi hai khối liên tiếp đạt, trần 75 phút (`--time-cap-s 4500`), chạy dưới `supervise`. Lệnh: `python runner.py supervise B3W0c --out runs/M2c --adam-iters 3000 --adam-lr 1e-4 --chunk 500 --max-iters 2500 --monitor-every 500 --stop-gates 2 --time-cap-s 4500`.

**M2′ ĐẠT khi (tất cả, tiêu chí M2 giữ nguyên):** (a) mọi loss và số đo hữu hạn, không sập quá một lần; (b) cổng đầy đủ đạt trên trọng số cuối; (c) `check_physics` 0 [WARN]; (d) ≤ 75 phút. **Trượt thì dừng, báo cáo, không chạy lại.**

**Chỉ ghi, không phải tiêu chí:** tỉ lệ điểm bị cắt; E_res và RMS residual theo dải nhiệt độ (so với B3 gốc: RMS > 1000 K 0.0398, và M2: 0.0635); bể chảy và chỉ báo ở 50/70/90%; so với M2 và B3. Tôi không đặt ngưỡng cho RMS vùng nóng vì thấy số của M2 trước khi viết; thước đo thực của chất lượng vùng nóng là M3 (độ ổn định theo tập điểm và ngân sách).

**Sau M2′:** nếu ĐẠT, **dừng và báo cáo** rồi hỏi người dùng trước khi chạy M3a, M3b, M4 (đăng ký 02, mục 7, giữ nguyên tiêu chí; chạy trên `B3W0c` với `--sobol-offset 10` cho M3b; hai bước huấn luyện mỗi bước tối đa 75 phút). M1 (vệt thường B3) dùng lại nếu S2.10(d) đạt. Nếu TRƯỢT: dừng, báo cáo.

## 5. Dự đoán (ghi trước, có thể sai)
- Tầng 2c đạt cả bảy lượt đầu: 65%.
- M2′ ĐẠT: 65%. Lý do: trọng số M2 sau khi cắt tại chỗ đã cho E_res +0.03 mJ và T_min = T_amb, nhưng huấn luyện lại với đầu ra cắt có thể đổi động học (vùng cắt không có gradient) và `check_physics` có thể có cảnh báo khác ở các phép 5–8 viết cho vệt đối xứng.
- Bể chảy M2′ ở 70% gần M2 (sâu 98 ± 3 µm, dài 525 ± 10, rộng 167 ± 3): 70%.
- RMS residual vùng > 1000 K của M2′ ≤ 0.0635 (không tệ hơn M2): 50%.
- Tổng GPU: tầng 2c khoảng 5 phút, M2′ khoảng 30 phút.

## 6. Giới hạn (khóa)
Cắt θ ≥ 0 sửa lỗi nhiệt độ âm ở vùng lạnh; nó **không** sửa lỗi trong vùng nóng, không thêm tính cách nhiệt của bột (hướng khác) và không phải hiệu chỉnh. Các giới hạn của 02 (mục 1, 11) giữ nguyên: chỉ báo là cận dưới, tương đối, một điều kiện, không có nghiệm đúng để so.

## 7. Kết quả tầng 2c (15:19–15:20, 30 s, mã `e3217ba`): **ĐẠT** cả bảy mục; ghi lúc 15:20

`legacy/ckpt` không đổi. Mã: `config.py` (`B3W0c`, `clamp_output`), `gates.py` (`band_diag`), `runner.py` (ghi `band_diag` vào `run.json`), `smoke/tier2c.py`.

| Mục | Kết quả |
|---|---|
| (a) | trọng số M2: không cắt min θ = −0.0204 (28.6% điểm âm); có cắt min 0; chênh lớn nhất nơi θ ≥ 0 bằng **0**; bằng 0 nơi θ < 0 |
| (b) | phần dư và gradient hữu hạn ở float32 và float64, dtype đúng |
| (c) canary B3 có cắt | rộng/sâu/dài 166.0 / 90.0 / 516.0 µm; tỉ lệ 0.805; E_res +0.26 mJ; T_min 298.000 K |
| (d) | chỉ báo B3 có cắt trùng `runs/M1/indicator.json` (lệch lớn nhất 0): M1 vẫn hợp lệ |
| (e) phân biệt | B3 PASS; B1k trượt (a, c: E_res +113.6 mJ, tỉ lệ 1.008); B1kc trượt (a: +44.7 mJ); B1f1.4 trượt (a: +20.8 mJ) |
| (f) | tag `B3W0` = `B3W0_fff2a3d4` (không đổi), tag `B3W0c` = `B3W0c_fff2a3d4`; hash `lpbf_pinn.py` đúng |
| (g) | `runner.py run B3W0c` tí hon: 24 s, mã thoát 0, 15 vòng, T_min 298.0 K, loss hữu hạn, tag `B3W0c_e9d36614` |

Ghi nhận: (1) tiền tố tag `B3W0c_` khác `B3W0_`, nhưng phần băm giống nhau ở hai tag (băm không chứa cờ cắt); tag khác nhau chỉ nhờ tên bậc. (2) Cắt làm E_res của ba checkpoint cũ **tăng** (+100.2 → +113.6; +31.4 → +44.7; +15.3 → +20.8 mJ), vì phần âm của vùng T < T_amb bị bỏ đi; nghĩa là cắt không làm cổng dễ qua hơn với các checkpoint cũ. (3) Dự đoán tầng 2c đạt cả bảy lượt đầu (65%): **đúng**.

Chạy M2′ ngay theo mục 4.

## 8. Kết quả M2′ (15:20–15:41, 21 phút, `runs/M2c/run.json`, tag `B3W0c_e90da666`): **TRƯỢT**; ghi lúc 15:43

Chạy đúng mục 4, không sập (`supervise` không thử lại). Adam 3,000 bước mất 435 s; **L-BFGS chỉ 1 khối, 269 vòng, 821 s, dừng `optimizer_stopped_early`** (loss 0.0063185 → 0.0063177, tức gần như không tiến; ở M2 không cắt, khối đầu giảm 0.0064 → 0.0054 trong 262 s).

| Tiêu chí M2′ | Kết quả | |
|---|---|---|
| (a) hữu hạn, không sập quá một lần | đạt | đạt |
| (b) cổng đầy đủ trên trọng số cuối | gate a: E_res **−16.1 mJ** (ngưỡng ±10) **trượt**; gate b: T_min 298.0 K (đạt theo cấu trúc); gate c: tỉ lệ 0.706 (đạt) | **trượt** |
| (c) `check_physics` 0 [WARN] | 0 cảnh báo | đạt |
| (d) ≤ 75 phút | 21 phút | đạt |

Diễn biến Adam (cổng nhẹ): E_res +10.5 → +10.7 → −4.3 → −12.7 → −15.1 → −16.0 mJ ở bước 500…3,000; tỉ lệ năng lượng 0.781 → 0.703. Giá trị cuối khối L-BFGS: E_res −16.1 mJ. Bể chảy ở 70%: dài/rộng/sâu = 511.9 / 166.3 (82.0 + 84.4) / 97.9 µm, độ xuyên 37.9 µm; chu trình tại đáy lớp bột: T_pk 2,950 K, lưu 479 µs, tốc độ nguội 1.267e6 K/s. So với M2: sâu −0.3, dài −13.6, rộng −0.4 µm; tốc độ nguội +12%. So với B3: sâu +7.1, dài −5.8 µm (hai lượt huấn luyện cùng cấu hình cho dài khác nhau 13.6 µm). **Các số này không được dùng như chỉ báo (M2′ trượt cổng).**

Chẩn đoán theo dải (`band_diag`, 2^18 điểm đều): 59% điểm bị cắt (E_res −0.6 mJ); dải 298–1000 K (36% điểm): **−14.8 mJ**, RMS 0.029; dải 1000–T_sol +0.4; ≥ T_sol −1.9 mJ; RMS residual vùng > 1000 K **0.0734** (B3 gốc 0.0398; M2 0.0635).

Chấm dự đoán (mục 5): tầng 2c đạt (65%) → đúng. **M2′ ĐẠT (65%) → sai.** Bể chảy gần M2 (sâu 98 ± 3, dài 525 ± 10, rộng 167 ± 3): sâu và rộng đúng, **dài sai** (511.9, lệch 13.6) → sai một phần. RMS vùng > 1000 K không tệ hơn M2 (50%) → **sai** (0.0734 so với 0.0635). Thời gian dự báo 30 phút, thực tế 21.

**Quyết định theo đăng ký:** dừng, báo cáo; không chạy M3, M4. M2 và M2′ đều là bản ghi TRƯỢT, không bản nào được sửa.

**Điều rút ra (số đo; chẩn đoán không huấn luyện, không phải mốc; `runs/M2c/diag_energy.json`)**
1. **Giả thuyết "chỉ do nhiệt độ âm" ở mục 1 quá mạnh.** Phép cắt tại chỗ trên trọng số M2 cho E_res +0.03 mJ, nhưng huấn luyện lại với đầu ra cắt **chuyển** phần nguồn âm sang dải 298–1000 K (−14.8 mJ), không làm nó biến mất. Nguồn âm cỡ −16…−26 mJ (khoảng 8–13% nạp 190 mJ) tồn tại ở cả hai lượt.
2. **Vị trí ổn định:** ở M2′ (và M2): vùng z 60–180 µm (−16.6 mJ; −17.7 ở M2), phía nêm y 0…100 µm (−14.9; −17.5), thời điểm muộn t > 0.5 (−17.6; −21.4), **phía sau laser dx < −150 µm (−17.1 mJ)**, trong đó dx < −600 µm chiếm −11.1 mJ (M2: −14.1). Đó là đuôi nguội phía nêm ở thời điểm muộn.
3. **Mật độ điểm huấn luyện thấp ở đuôi:** dx < −600 µm có 4.9% điểm neo trên 17.8% thể tích (0.28× mật độ đều); tổ hợp "đuôi, t > 0.5, y > 0, z 60–180 µm" chỉ 0.5% điểm trên 4.2% thể tích (**0.13×**). Vùng gần laser (−150…100 µm) dày 3.35×. Phù hợp với giả thuyết lệch lấy mẫu, chưa phải bằng chứng nhân quả.
4. **L-BFGS không tiến khi có cắt** (gần như đứng yên, chạy 3× chậm hơn mỗi vòng, tự dừng sớm). Nguyên nhân chưa kiểm; giả thuyết: gradient không liên tục của max làm tìm kiếm đường thất bại.
5. Phép cắt vẫn là một tính chất đúng của nghiệm và giữ T_min = T_amb; nó không đủ để qua gate (a).

## 9. Sửa đổi 03a (16:29, trước khi viết mã và chạy): thí nghiệm A1 — kiểm giả thuyết lấy mẫu ở đuôi

Người dùng chọn A1 sau khi xem mục 8. **Đây là thí nghiệm chẩn đoán, không phải lượt M2/M2′**: không thay kết luận TRƯỢT của M2 và M2′, không dùng để tuyên bố đạt cổng; nó chỉ trả lời "thêm điểm huấn luyện ở đuôi nguội phía sau laser có làm nguồn âm ở đó co lại không".

**Giả thuyết H_s:** nguồn âm cỡ −16 mJ của M2′ ở đuôi (dx < −150 µm phía sau laser, t muộn, phía nêm) là hệ quả của mật độ điểm huấn luyện thấp ở đó (0.28× mật độ đều ở dx < −600 µm; 0.13× ở tổ hợp đuôi, t > 0.5, y > 0, z 60–180 µm). Giả thuyết đối lập: sức chứa mạng (6 × 24) hoặc độ khó của bài toán, thêm điểm không giúp.

**Thiết kế (khởi động chung từ trọng số M2′ `runs/M2c/B3W0c_e90da666.pt`, bậc `B3W0c`, đầu ra cắt, Adam 1,000 bước lr 1e-4 toàn lô, `compile` một lần, float32):**
- **Nhánh C (đối chứng):** tập điểm neo của M2′ không đổi (364,544 điểm).
- **Nhánh T (thêm điểm đuôi):** thêm **2^17 = 131,072 điểm đều** trong vùng R = {x − x_laser(t) < −150 µm} (lấy bằng loại bỏ mẫu từ phân bố đều trên hộp x thời gian, seed 2026, y ∈ (−Ly/2, Ly/2), z ∈ [0, Lz) đều), nối vào tập neo; mọi thứ khác như C. Mật độ trong R sau khi thêm được báo cáo (dự tính khoảng 1× mật độ trung bình ở dx < −600 µm).
- Điểm đánh giá độc lập với điểm huấn luyện: `probe` (seed 101) và chẩn đoán theo vùng 2^18 điểm (seed 777), giống mục 8.
- Đo ở bước 0 (= M2′), và ở cuối mỗi nhánh (bước 1,000): cổng nhẹ mỗi 250 bước (chỉ ghi); ở cuối: `probe` đầy đủ (E_res, T_min, tỉ lệ), E_res theo vùng (dx, t, y, z, dải T) và RMS residual vùng > 1000 K, bể chảy và chỉ báo ở 70%.
- Mã: `experiments/a1_tail.py`; kết quả `runs/A1/report.json`. Trần GPU 15 phút (dừng ở 22.5). Không sửa `legacy/`; trọng số hai nhánh lưu trong `runs/A1/` (không vào `legacy/ckpt`).

**Quy tắc quyết định (khóa trước khi chạy).** Ký hiệu: E_full = E_res của `probe` đầy đủ ở cuối nhánh (mJ, âm là nguồn âm); E_tail = E_res trong vùng dx < −600 µm theo chẩn đoán 2^18 điểm (M2′ ở bước 0: −11.1 mJ; tổng 2^18 điểm −16.9 mJ; `probe` −16.1 mJ).
- **H_s được ủng hộ** nếu E_full(T) − E_full(C) ≥ **+8 mJ** (T gần 0 hơn C) **và** E_tail(T) ≥ **−5.5 mJ** (co ít nhất một nửa so với −11.1).
- **H_s không được ủng hộ** nếu E_full(T) − E_full(C) < **+3 mJ**.
- Còn lại: **không kết luận** (báo cả hai số, không suy diễn).
- Ngưỡng 8 và 3 mJ lớn hơn nhiễu đánh giá (sai số chuẩn E_res của `probe` khoảng 1 mJ trên các checkpoint đã đo). Các ngưỡng này là **[ĐỀ XUẤT]** của tôi.
- Chỉ ghi, không thuộc quy tắc: có qua gate (a) (|E_full| ≤ 10 mJ) không; RMS vùng > 1000 K (M2′ 0.0734) và bể chảy có xấu đi không (cảnh báo nếu RMS tăng > 20% hoặc sâu/dài đổi > 5 µm so với bước 0).

**Dự đoán (ghi trước):** H_s được ủng hộ 40%; không được ủng hộ 35%; không kết luận 25%. Lý do giữ không chắc: bốn phần năm sự cải thiện có thể bị giới hạn bởi sức chứa mạng nhỏ và bởi 1,000 bước ngắn. Nhánh C: E_full thay đổi không quá 3 mJ so với bước 0 (65%), vì Adam của M2′ vẫn đang trôi về phía âm hơn ở những bước cuối (−12.7 → −15.1 → −16.0).

**Nếu H_s được ủng hộ:** bước kế tiếp là đăng ký 03b (lấy mẫu có mật độ đều theo thể tích hoặc có trọng số loss) và chạy lại M2′ đầy đủ; **không** tự chạy. Nếu không được ủng hộ: dừng, thảo luận hướng khác (sức chứa mạng, hướng bột/hướng 4 trên vệt thường).

## 10. Kết quả A1 (16:31–16:47, 15.5 phút GPU, mã `a1_tail.py`, `runs/A1/report.json`): **H_s KHÔNG được ủng hộ**; ghi lúc 16:48

`legacy/ckpt` không đổi. Điểm thêm: 131,072 điểm đều trong vùng dx < −150 µm; mật độ tương đối ở dx < −600 µm tăng 0.28× → **0.82×** (dự tính trước khoảng 1×; số đo thật thấp hơn), ở −600…−150 µm 1.12× → 1.43×, gần laser giảm 3.35× → 2.46×.

| | Bước 0 (M2′) | Nhánh C (đối chứng) | Nhánh T (thêm điểm đuôi) |
|---|---|---|---|
| E_res `probe` đầy đủ (mJ, ± sai số chuẩn) | −16.1 ± 1.0 | −17.2 ± 1.0 | **−19.3 ± 0.9** |
| E_res vùng dx < −600 µm (mJ) | −11.1 | −11.5 | **−11.0** |
| E_res vùng −600…−150 µm (mJ) | −6.0 | −5.6 | −7.3 |
| RMS residual vùng > 1000 K | 0.0734 | 0.0716 | 0.0716 |
| bể chảy rộng/sâu/dài (µm, lưới 2 µm) | 164/98/512 | 164/98/512 | 164/98/510 |
| loss cuối [L_PDE, L_top] | 6.32e-3 | 5.98e-3 | 4.46e-3 (tập điểm khác) |
| thời gian huấn luyện 1,000 bước | | 168 s | **754 s** (1.36× điểm, 4.5× thời gian) |

**Quyết định theo quy tắc khóa (mục 9):** E_full(T) − E_full(C) = **−2.2 mJ** < +3 mJ → **H_s không được ủng hộ**. E_tail(T) = −11.0 mJ (cần ≥ −5.5 để ủng hộ): vùng đuôi không co. Gate (a) vẫn trượt ở cả hai nhánh (|E_res| > 10); gate (b) đạt theo cấu trúc, gate (c) đạt (tỉ lệ 0.70). Bể chảy và RMS vùng nóng không xấu đi.

Chấm dự đoán (mục 9): H_s không được ủng hộ (35%) → **xảy ra**; nhánh C đổi E_full không quá 3 mJ (65%) → đúng (−1.1 mJ).

**Giới hạn của thí nghiệm (phải ghi):** nhánh C chỉ giảm loss 5% và E_res đổi −1.1 mJ sau 1,000 bước Adam lr 1e-4, nên thí nghiệm có sức phát hiện thấp với thay đổi chậm; nó trả lời "thêm điểm đuôi trong 1,000 bước Adam không co nguồn âm", không loại trừ hoàn toàn tác dụng của lấy mẫu qua huấn luyện dài. Thời gian nhánh T 4.5× với 1.36× điểm: nghi bộ nhớ GPU gần đầy (15.9 / 16.4 GB), chưa kiểm. **Đính chính:** câu "bốn phần năm sự cải thiện có thể bị giới hạn…" ở dự đoán mục 9 là câu viết sai; ý định là "mức cải thiện có thể bị giới hạn bởi sức chứa mạng nhỏ và bởi 1,000 bước ngắn"; quy tắc quyết định và ngưỡng không đổi.

**Chẩn đoán bổ sung (không huấn luyện, không phải mốc; E_res theo vật liệu `s` của `powder_index`, 2^18 điểm đều seed 777):**

| Trường | rắn (s < 0.01) | mặt phân cách | bột (s > 0.99) |
|---|---|---|---|
| B3 @ B3 (vệt thường) | 71.4% thể tích: **−8.07 mJ** | 7.2%: −0.27 | 21.4%: +0.49 |
| M2′ @ nêm | 19.8%: −1.39 | 7.5%: −3.33 | **72.7%: −12.13** (dải 298–1000 K: −11.55; RMS 0.024) |

Ở nêm, vùng "bột" gồm lớp 60 µm trên cùng **và** vùng dưới phần nhô (tam giác y < 180 − z); phần rắn chỉ còn là góc nêm (19.8% thể tích). Nguồn âm của M2′ nằm chủ yếu ở vùng bột/dưới phần nhô, trong dải 298–1000 K, với RMS residual 0.024 (nhỏ so với 0.073 ở vùng nóng) trên 73% thể tích; ở B3 vệt thường nó nằm ở nền rắn.

**Giả thuyết còn lại (chưa kiểm):** E_res là hàm tuyến tính của phần dư (trung bình của r), còn loss là bình phương (trung bình r²). Sai số có dấu hệ thống nhỏ (|r| ~ 0.02–0.03) trên thể tích lớn bị loss bình phương phạt yếu (0.024² so với 0.073² ở vùng nóng) nhưng cộng thành −12 mJ trong E_res. Nếu đúng, lấy thêm điểm (A1) không đủ, cần thêm số hạng loss theo cân bằng năng lượng (trung bình phần dư), một điều kiện vật lý chính xác của PDE; đánh đổi: cổng E_res khi đó là một phần của mục tiêu huấn luyện, nên mất vai trò kiểm độc lập.

## 11. Sửa đổi 03b (16:51, trước khi viết mã và chạy): hướng B1 — số hạng cân bằng năng lượng trong loss

Người dùng chọn B1 sau mục 10. Giả thuyết **H_c:** E_res là trung bình tuyến tính của phần dư r, còn loss PDE là trung bình r²; sai số hệ thống nhỏ (|r| khoảng 0.02) trên thể tích lớn ở vùng bột/dưới phần nhô bị loss phạt yếu nhưng cộng thành −12…−16 mJ. Nếu đúng, thêm một số hạng loss phạt **phần dư trung bình theo vùng** sẽ đưa E_res về trong ±10 mJ trên điểm độc lập, mà không làm xấu vùng nóng hay bể chảy.

**Đánh đổi phải khai trước.** Số hạng này huấn luyện trực tiếp lên đại lượng mà cổng (gate a) đo, nên gate (a) không còn là kiểm **độc lập** của "không có nguồn giả" nếu chỉ đo trên điểm huấn luyện. Biện pháp: (i) số hạng dùng **mẫu đều mới ở mỗi bước** (không phải tập cố định), (ii) đánh giá trên các tập điểm độc lập (`probe`, seed 101; chẩn đoán 2^18 điểm, seed 777) không dính tới huấn luyện, (iii) tiêu chí thêm: cân bằng theo từng loại vật liệu trên điểm độc lập (chống triệt tiêu giữa các vùng), (iv) bể chảy, RMS vùng nóng và `check_physics` phải không xấu đi. Số hạng này là hệ quả chính xác của PDE (tích phân PDE trên một vùng cho cân bằng năng lượng vùng), không phải dữ liệu.

**Số hạng (mới, thêm vào loss của bậc `B3W0c`; mã `energy.py`):**
L_E = Σ_c e_c², với e_c = E_c / E_in, c chạy trên 6 lớp = {rắn (s < 0.01), mặt phân cách (0.01 ≤ s ≤ 0.99), bột (s > 0.99)} × {t < t_end/2, t ≥ t_end/2}; E_c = (1/N) Σ_{i thuộc c} r_i · E_r (E_r = ρ c_p T_c L_c³ L_x L_y L_z t_end [J], như `probe`); N = 2^15 điểm đều trong hộp (x, y, z, t), **lấy mới mỗi bước**; E_in = 0.18999 J (laser nạp). Loss tổng: L_PDE + L_top + λ·L_E (gắn qua `loss_weights` của DeepXDE, lớp con `TimePDE` thêm một số hạng loss; giá trị L_PDE và L_top không đổi so với `make_data` — kiểm ở mốc S2.12(a)). Khi M2′ ở bước 0: E_bột/E_in ≈ −0.064, nên L_E ≈ 0.005 cùng bậc L_PDE (6.3e-3) ở λ = 1.

**Mốc kiểm trước khi chạy (S2.12).** (a) Lớp con `TimePDE` với λ = 0 cho L_PDE và L_top ở bước 0 trùng `L.make_data` (sai ≤ 1e-6 tương đối: M2′ cho [6.32e-3, 1.04e-4]); (b) L_E tại trọng số M2′ cho Σ_c E_c tổng khớp E_res đo bằng `probe` trong 3 mJ (ước lượng độc lập, cùng dấu) và gradient trọng số hữu hạn; (c) 10 bước Adam λ = 1 chạy hết, loss hữu hạn, không NaN. Trượt thì dừng, báo cáo.

**Thí nghiệm B1 (chẩn đoán, giống A1: không phải lượt M2).** Khởi động từ trọng số M2′ (`runs/M2c/B3W0c_e90da666.pt`), bậc `B3W0c` (đầu ra cắt), Adam **1,000 bước** lr 1e-4 toàn lô, hai nhánh **λ = 1** và **λ = 10**; đối chứng là nhánh C của A1 (không số hạng; E_full −17.2 mJ ở 1,000 bước) vì cùng điểm khởi đầu, cùng số bước, cùng lr. Đo ở cuối: `probe` đầy đủ (E_res, T_min, tỉ lệ), E_res theo vùng (dx, t, y, z, dải T) và theo vật liệu s (2^18 điểm, seed 777), RMS vùng > 1000 K, bể chảy và chỉ báo ở 50/70/90%, `check_physics` (không có ý nghĩa như cổng ở đây nhưng ghi số [WARN]); cổng nhẹ mỗi 250 bước. Mã `experiments/b1_energy.py`, kết quả `runs/B1/report.json`, trần GPU 15 phút (dừng ở 22.5).

**Quy tắc quyết định (khóa trước khi chạy).** Một nhánh λ **thành công** khi đạt **tất cả**, trên điểm độc lập:
1. |E_res| của `probe` đầy đủ ≤ **10 mJ** (gate a).
2. Mỗi lớp vật liệu (rắn, mặt phân cách, bột; gộp hai nửa thời gian), E_res theo chẩn đoán 2^18 điểm: |E_c| ≤ **7 mJ**.
3. RMS vùng > 1000 K ≤ **1.2 × bước 0** (0.0734 → ≤ 0.0881).
4. Bể chảy ở 70% (rộng, sâu, dài, lưới 2.5 µm) lệch ≤ **5 µm** so với bước 0 (164/98/512 µm theo lưới 2 µm; so cùng chỉ báo 2.5 µm: 166.3 / 97.9 / 511.9).
5. Gate (c) vẫn đạt (tỉ lệ ≤ 1); loss hữu hạn.
**B1 được ủng hộ** nếu ít nhất một nhánh thành công; khi cả hai thành công chọn λ **nhỏ hơn** làm λ* (ít can thiệp hơn). **B1 không được ủng hộ** nếu không nhánh nào đạt tiêu chí 1. Nếu đạt 1 nhưng trượt 2–4: **không kết luận** (báo số, thảo luận; không suy diễn). Ngưỡng 7 mJ, 1.2×, 5 µm là **[ĐỀ XUẤT]**.

**Bước sau (không chạy tự động):** nếu B1 được ủng hộ, viết đăng ký 03c cho M2″ (chạy lại từ B3 với số hạng và λ*, cùng tiêu chí M2) — lưu ý ở M2′ L-BFGS không tiến khi có cắt nên M2″ dự kiến chỉ dùng Adam; hỏi người dùng trước. Nếu không được ủng hộ: dừng, thảo luận B2 (hướng 4 trên vệt thường).

**Dự đoán (ghi trước):** S2.12 đạt cả ba lượt đầu 80%. B1 được ủng hộ 50% (λ = 10 đạt 45%, λ = 1 đạt 25%). Rủi ro chính: 1,000 bước Adam lr 1e-4 đủ để dời E_res về 0 (A1 nhánh C chỉ đổi −1.1 mJ không có số hạng, nên chưa rõ tốc độ); và số hạng có thể đẩy sai số sang chỗ khác (lớp thời gian, vùng nóng). Gate (a) về trong ±10 mJ 60% (ít nhất một nhánh); tiêu chí 2 đạt khi đã đạt 1: 80%; vùng nóng không xấu quá 1.2×: 70%. Tổng GPU khoảng 12 phút.

## 12. Kết quả S2.12 (6 s, mã `energy.py`, `smoke/tier2d.py`): **ĐẠT** cả ba mục; ghi lúc 16:53

(a) lớp con `TimePDE` với λ = 0 cho [L_PDE, L_top] = [0.0063181, 0.00010414] trùng `L.make_data` (lệch 0); (b) ước lượng E_res của số hạng (2^17 điểm đều mới) = −17.09 mJ so với `probe` −16.09 mJ (lệch 1.0, cùng dấu); sáu lớp (rắn, phân cách, bột × nửa đầu, nửa sau): +1.16, −2.61, −0.44, −2.72, −0.19, **−12.29** mJ (bột, nửa sau thời gian chiếm phần lớn); L_E = 5.17e-3, gradient hữu hạn; (c) 10 bước Adam λ = 1: loss hữu hạn, ba số hạng (buổi đầu [6.32e-3, 1.04e-4, 4.50e-3], cuối [1.20e-2, 5.1e-4, 9.3e-4]). Chạy thử đường ống B1 với 8 bước (không dùng kết quả) cho thấy E_res đổi nhanh theo số hạng (−16 → −1…−5 mJ chỉ sau 8 bước) và RMS vùng nóng nhảy tạm 0.10–0.12 vì Adam khởi động lại (như ở A1: loss tăng ở vài bước đầu); tiêu chí 3 chỉ được tính ở bước 1,000. Chạy B1 chính thức ngay.

## 13. Kết quả B1 (16:53–16:59, 6 phút GPU, `experiments/b1_energy.py`, `runs/B1/report.json`): theo quy tắc khóa **KHÔNG KẾT LUẬN**; ghi lúc 17:00

`legacy/ckpt` không đổi. Adam 1,000 bước từ trọng số M2′ (bước 0 = M2′), đo trên điểm độc lập (`probe` seed 101; chẩn đoán 2^18 điểm seed 777; chỉ báo lưới 2.5 µm).

| | Bước 0 (M2′) | λ = 1 | λ = 10 |
|---|---|---|---|
| E_res `probe` (mJ, ± sai số chuẩn ≈ 1.0) | −16.1 | **+0.2** | **+3.1** |
| gate a / b / c | trượt / đạt / đạt | đạt / đạt / đạt | đạt / đạt / đạt |
| tỉ lệ (tích + bay hơi)/nạp | 0.706 | 0.774 | 0.785 |
| E_res theo vật liệu: rắn / phân cách / bột (mJ) | −1.4 / −3.3 / −12.1 | +1.1 / −1.1 / −0.4 | +0.9 / −0.1 / +1.3 |
| E_res dải T: 298–1000 K / ≥ 1000 K (mJ) | −14.8 / −1.4 | −1.9 / +2.2 | −4.4 / **+7.2** |
| E_res vùng đuôi dx < −600 µm (mJ) | −11.1 | +0.7 | −0.4 |
| RMS residual vùng > 1000 K | 0.0734 | 0.0722 | 0.0764 |
| `check_physics` [WARN] | 0 | 0 | 0 |
| chỉ báo ở 70%: dài / rộng (W₊ + W₋) / sâu (µm) | 511.9 / 166.3 / 97.9 | 517.1 / 166.5 / 98.0 | **535.0** / 167.1 / 98.4 |
| tốc độ nguội tại đáy lớp bột (K/s) | 1.267e6 | 1.206e6 | 1.057e6 |
| bể chảy dài (lưới 2 µm) | 512 | 516 | 534 |
| thời gian huấn luyện | | 174 s | 172 s |

Tiêu chí (mục 11): λ = 1: (1) đạt, (2) đạt (|E_c| ≤ 1.1), (3) đạt (0.0722 ≤ 0.0881), **(4) trượt: dài lệch +5.2 µm (> 5)**, rộng +0.2, sâu +0.1, (5) đạt. λ = 10: (1) đạt, (2) đạt, (3) đạt, **(4) trượt: dài lệch +23.1 µm**, (5) đạt. **Quy tắc khóa:** đạt (1) nhưng trượt (4) → **không kết luận** (báo số, thảo luận, không suy diễn). Cả hai nhánh trượt (4) nên không có λ* theo quy tắc.

Chấm dự đoán (mục 11): B1 được ủng hộ 50% → **không** (theo quy tắc, "không kết luận"); gate (a) về ±10 mJ ở ít nhất một nhánh 60% → **đúng (cả hai)**; λ = 10 thành công 45% → sai (dài +23 µm); λ = 1 thành công 25% → sai, trượt 0.2 µm; (2) đạt khi (1) đạt 80% → đúng; vùng nóng không xấu quá 1.2× 70% → đúng. Thời gian dự báo 12 phút, thực tế 6.

**Số đo cần ghi, không suy diễn ngoài số liệu.**
1. H_c được ủng hộ cho phần E_res: số hạng cân bằng năng lượng đưa E_res từ −16.1 về +0.2 mJ (λ = 1) và cân bằng theo từng vật liệu trên điểm **độc lập** (không dùng trong huấn luyện), nên không chỉ là khớp lên mẫu; loss PDE trên điểm huấn luyện gần như không đổi (6.3e-3 → 6.25e-3 ở λ = 1), nghĩa là E_res là hệ quả của sai số hệ thống nhỏ mà loss bình phương phạt yếu.
2. Chiều dài bể chảy đổi theo: +5.2 µm (λ = 1), +23 µm (λ = 10). Với λ = 1 chiều dài 517.1 µm gần B3 vệt thường (517.7 µm), còn M2 525.5 µm và M2′ 511.9 µm (hai lượt huấn luyện cùng cấu hình khác nhau 13.6 µm). Dài là đại lượng nhạy: biên đuôi bể nằm ở nơi T ≈ T_sol với gradient nhẹ. Tiêu chí (4) (≤ 5 µm so với bước 0) trượt λ = 1 chỉ 0.2 µm, nhưng giá trị đó nằm trong khoảng dao động đã thấy giữa các lượt huấn luyện (13.6 µm). **Quy tắc không bị sửa.**
3. λ = 10 quá mạnh: E_res dải ≥ 1000 K +7.2 mJ (nguồn dương ở vùng nóng), chiều dài +23 µm, tốc độ nguội −16%, RMS vùng nóng 0.0764 (cao nhất); loss L_PDE tăng 6.3e-3 → 7.0e-3. λ = 1 cân bằng hơn: ≥ 1000 K +2.2 mJ, RMS 0.0722.
4. ΔD sâu so với vệt thường B3 (M1: 90.8 µm): +7.2 µm (λ = 1), +7.6 (λ = 10), khớp ước lượng trước (+5…+20 µm); **chưa phải chỉ báo** vì chưa có lượt qua cổng chính thức (M2″) và chưa có M3.
5. Phạm vi: đây là tinh chỉnh 1,000 bước từ trọng số M2′, không phải lượt huấn luyện đầy đủ từ B3; cổng (a) đạt ở đây không thay thế M2″.

## 14. Kiểm toán cân bằng năng lượng độc lập và **đính chính** các kết luận ở mục 1, 2, 8, 13 (17:07; chẩn đoán không huấn luyện, không phải mốc)

**Phép đo.** Đồng nhất thức E_id = E_lưu_trữ(t_end) + E_qua_5_vách + E_bay_hơi − E_nạp (E_nạp = ∫∫ q bằng Monte Carlo 2^20 điểm: 186.9–187.3 mJ so với 190.0 mJ của `check_process`). Thông lượng qua vách đo bằng đạo hàm bậc một tại 2^17 điểm mỗi mặt, lùi vào trong 2 µm (kiểm 8 µm: khác ≤ 1.1 mJ). Với trường **không có điểm gãy**, E_id phải bằng E_res (tích phân phần dư PDE) theo định lý phân kỳ. Script thử nghiệm trong thư mục tạm của phiên (không vào kho); số đo ở dưới.

| Trường | E_res trực tiếp (`probe`, mJ) | E_id (mJ) | lệch | lưu trữ / qua vách / bay hơi (mJ) |
|---|---|---|---|---|
| B3 @ B3, không cắt | −7.4 | −5.6 | +1.8 | 117.0 / 28.7 / 36.0 |
| M2 @ nêm, không cắt | −23.5 | **−25.0** | −1.5 | 107.6 / 18.3 / 36.1 |
| B3 @ B3, **cắt** sau huấn luyện | +0.3 | −3.2 | −3.5 | |
| M2 (**cùng trọng số**), **cắt** | **+1.3** | **−14.7** | **−16.0** | 108.0 / 28.0 / 36.1 |
| M2′ (huấn luyện có cắt) | −16.1 | **−34.8** | −18.7 | 98.3 / 18.1 / 35.7 |
| B1 λ = 1 | +0.2 | **−18.4** | −18.6 | 111.5 / 21.4 / 35.6 |
| B1 λ = 10 | +3.1 | **−19.5** | −22.6 | 113.4 / 18.3 / 35.7 |

**Kết luận của kiểm toán (số đo):**
1. Với trường không cắt, phép đo trực tiếp E_res **đúng** (khớp E_id trong 2 mJ, ở B3 và M2). Nguồn âm của M2 là thật: −25 mJ, 13% năng lượng nạp.
2. **Phép cắt θ ≥ 0 che lỗi chứ không sửa lỗi.** Trên **cùng trọng số M2**, cắt làm E_res trực tiếp từ −23.5 xuống +1.3 mJ nhưng E_id chỉ đổi −25.0 → −14.7 mJ. Phần chênh (khoảng −16 mJ) là một **nguồn âm phân bố trên mặt gãy** của max(·, 0): đạo hàm bậc hai của hàm cắt bằng 0 hầu như mọi nơi, nên phần dư theo điểm (và vì vậy cổng E_res lẫn loss) **không thấy** bước nhảy thông lượng ở đó, trong khi năng lượng thực bị mất ở bề mặt đó. M2′ và B1 mang cùng khiếm khuyết (lệch −18.6 đến −22.6 mJ).
3. Hệ quả: **"B1 đưa E_res về +0.2 mJ" chủ yếu là khớp đại lượng đo, không phải khôi phục bảo toàn năng lượng**: E_id của B1 λ = 1 vẫn −18.4 mJ (10% nạp). Thứ tự theo E_id (trung thực): B3 vệt thường −5.6 (đạt ±10); B1 λ = 1 −18.4; B1 λ = 10 −19.5; M2 −25.0; M2′ −34.8. **Không trường nào trên nêm đạt ±10 mJ theo E_id.**
4. **Đính chính phát biểu của tôi.** (a) Mục 2: "đạo hàm bậc hai của max bằng 0 gần như mọi nơi nên phần dư không thấy điểm gãy" được viết như một ưu điểm; thực ra đó là khuyết điểm (không thấy = không phạt). (b) Mục 1 và mục 8: "cổng (a) và (b) gần như chỉ đo lỗi nhiệt độ âm", "cắt là tính chất đúng của nghiệm" chỉ đúng nửa: nhiệt độ âm là lỗi thật, nhưng cắt không bảo toàn thông lượng nên không phải cách sửa. (c) Mục 13: B1 không phải bằng chứng cho H_c theo nghĩa bảo toàn năng lượng; nó chỉ cho thấy phần dư trung bình theo điểm có thể bị ép về 0 mà loss PDE không đổi. Các kết luận TRƯỢT của M2 và M2′ **không đổi** (E_id còn xấu hơn); không kết quả nào được sửa thành đạt.

**Quan sát bổ sung (từ số đã có, chưa phải mốc).** Mức |E_id| (hoặc E_res không cắt) tỉ lệ gần tuyến tính với loss PDE huấn luyện cuối cùng trong sáu checkpoint khác nhau về vật lý: B3 L_PDE 1.05e-3 → 7.4 mJ; B1f1.4 1.82e-3 → 15.3; B1kc 4.62e-3 → 31.4; B1k 1.68e-2 → 100.2; M2 4.30e-3 → 25.0 (E_id); M2′ 6.32e-3 → 34.8 (E_id); tức khoảng **6–8 mJ cho mỗi 1e-3** (B1 λ = 1: 2.9 mJ cho mỗi 1e-3, nhờ số hạng năng lượng). Gate (a) ±10 mJ tương ứng L_PDE khoảng 1.4e-3 theo quan hệ này. Đây là quan hệ thực nghiệm trên 6 điểm, chưa kiểm nhân quả; nếu đúng, khiếm khuyết bảo toàn là hệ quả của mức hội tụ của loss (ở nêm, M2 dừng ở 4.3e-3 sau 5,500 bước, B3 dừng ở 1.05e-3 sau 21,500 bước) chứ không phải lỗi riêng của hình học hay đầu ra.

## 15. Sửa đổi 03c (17:14, trước khi viết mã và chạy): hướng 1 (cổng E_id trung thực) cộng hướng 2 (phạt dương thay cắt cứng)

Người dùng chọn hướng 1 và hướng 2 sau mục 14. Bản ghi `B3W0c` (cắt cứng), M2′, A1, B1 giữ nguyên như bản ghi; từ đây không dùng cắt cứng cho lượt mới.

**Hướng 1 — cổng trung thực.** Định nghĩa E_id như mục 14: E_id = E_lưu_trữ(t_end) + E_qua_5_vách + E_bay_hơi − E_nạp. E_nạp = ∫∫ q (Monte Carlo); E_lưu_trữ và E_bay_hơi từ `probe` (đầy đủ) hoặc cổng nhẹ; thông lượng qua vách −k ∂T/∂n_ngoài, đạo hàm bậc một tại các điểm cách vách 2 µm (2^17 điểm mỗi mặt ở bản đầy đủ, 2^15 ở bản nhẹ; E_nạp 2^20 / 2^18 điểm). **Gate (a) nay = (|E_res| ≤ 10 mJ) VÀ (|E_id| ≤ 10 mJ)**; ngưỡng không đổi. Lý do: phép đo E_res theo điểm chỉ đúng khi trường không có điểm gãy (mục 14); E_id không dùng đạo hàm bậc hai nên không bị đánh lừa. Áp dụng cho mọi lượt từ đây; tính lại cho các lượt cũ không làm đổi kết luận nào (M2: E_id −25.0, M2′: −34.8; cả hai vẫn TRƯỢT; B3 vệt thường −5.6 đạt). Sai số Monte Carlo của E_id khoảng ±2 mJ.

**Hướng 2 — phạt dương.** Bậc \`B3W0\` (**không cắt**, khởi động từ trọng số M2 \`runs/M2/B3W0_e90da666.pt\`). Thêm hai số hạng vào loss (gắn qua \`loss_weights\` của DeepXDE, mẫu đều mới mỗi bước như L_E): λ_E·L_E (như mục 11, 6 lớp vật liệu × nửa thời gian, nay hợp lệ vì trường trơn) và **λ_P·L_P, L_P = mean ReLU(−θ)²** trên N = 2^15 điểm đều, θ = (T − T_amb)/T_c. Hàm này trơn hai lần phần dư theo điểm nên **không tạo nguồn ẩn**, và gate (b) (T_min) vẫn **là phép kiểm có thông tin** (không qua theo cấu trúc). Không chọn biến đổi dương mượt (softplus): gate (b) qua theo cấu trúc, và lớp chuyển rất mỏng khó lấy mẫu đủ nên có nguy cơ lặp lại nguồn ẩn (chưa thử; ghi lại như phương án).

**Mốc test tầng 2e (S2.13, chạy trước thí nghiệm; trượt thì dừng, báo cáo):**
| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | E_id (bản đầy đủ) trên B3 vệt thường và trên M2 (không cắt) | B3: −5.6 ± 2 mJ; M2: −25.0 ± 2 mJ (số mục 14); lùi 8 µm thay 2 µm khác ≤ 2 mJ |
| (b) | Hai đường đo khớp với trường không có điểm gãy | |E_res − E_id| ≤ 3 mJ cho B3 và M2 |
| (c) | **Cổng phát hiện nguồn ẩn:** M2 sau khi cắt (cùng trọng số) | E_res đo được |E_res| ≤ 10 mJ nhưng |E_id| > 10 mJ, tức cổng (a) mới **TRƯỢT** |
| (d) | Cổng mới trên bốn checkpoint cũ | B3 PASS cả (a); B1k, B1kc, B1f1.4 TRƯỢT (a) |
| (e) | Số hạng L_P: giá trị, gradient hữu hạn; L_P(M2) khớp tính độc lập trên 2^17 điểm khác (tỉ lệ 0.8–1.25); ba số hạng phụ (L_PDE, L_top không đổi khi λ = 0 như S2.12a) | đúng |
| (f) | Hồi quy: tầng 2 (S2.1–S2.4) và S2.12 chạy lại với mã mới | đạt |
Dự đoán S2.13 đạt cả sáu 65% (rủi ro: (a), (b) lệch quá 2–3 mJ vì Monte Carlo).

**Thí nghiệm C1 (chẩn đoán, giống A1/B1: không phải lượt M2).** Từ trọng số M2 (\`B3W0\`, không cắt), Adam 1,500 bước lr 1e-4 toàn lô, `compile` một lần, ba nhánh: **C0** (đối chứng, λ_E = λ_P = 0), **C1a** (λ_E = 1, λ_P = 100), **C1b** (λ_E = 1, λ_P = 1000). λ_P chọn để λ_P·L_P cùng bậc L_E ở bước 0 (L_P(M2) khoảng 1e-5 nên λ_P = 100 cho 1e-3, λ_P = 1000 cho 1e-2). Đo trên điểm độc lập như B1: \`probe\` đầy đủ, E_id, hụt T_min, chẩn đoán theo vùng và theo vật liệu, RMS vùng > 1000 K, chỉ báo ở 50/70/90%, \`check_physics\`. Mã \`experiments/c1_positivity.py\`, kết quả \`runs/C1/report.json\`; trần GPU 25 phút (dừng ở 37).

**Quy tắc quyết định (khóa trước khi chạy).** Một nhánh **thành công** khi đạt **tất cả**, trên điểm độc lập:
1. |E_res| ≤ 10 mJ **và** |E_id| ≤ 10 mJ.
2. Hụt T_min ≤ 1% biên độ (gate b, có thông tin).
3. Gate (c): tỉ lệ (tích + bay hơi)/nạp ≤ 1; loss hữu hạn.
4. Mỗi lớp vật liệu (gộp hai nửa thời gian) |E_c| ≤ 7 mJ.
5. RMS vùng > 1000 K ≤ 1.2 × của M2 (0.0635 → ≤ 0.0762).
6. Bể chảy ở 70%: **sâu và rộng** (W₊ + W₋) lệch ≤ 5 µm so với M2; **dài chỉ ghi** (xem đính chính).
7. \`check_physics\` 0 [WARN].
**C1 được ủng hộ** nếu ít nhất một nhánh đạt (1)–(7) **và** C0 trượt (1) hoặc (2) (để quy hiệu quả cho số hạng mới); khi cả hai nhánh đạt, λ* = nhánh có λ_P nhỏ hơn. **C1 không được ủng hộ** nếu không nhánh nào đạt (1). Còn lại: **không kết luận**. Ngưỡng 7 mJ, 1.2×, 5 µm [ĐỀ XUẤT] giữ như 03b.

**Đính chính tiêu chí chiều dài (03b, tiêu chí 4).** Ở B1 tôi đặt "dài lệch ≤ 5 µm so với bước 0" làm điều kiện thành công. Sai về thiết kế: chiều dài là đại lượng nhạy (M2 và M2′, hai lượt cùng cấu hình, khác nhau 13.6 µm; B3 vệt thường 517.7 µm giữa các thời điểm 482–530 µm), và khi sửa đúng cân bằng năng lượng thì đuôi bể **phải** đổi (B1 λ = 1: +5.2 µm). Điều kiện đó loại trừ đúng điều ta muốn đạt. Từ C1 trở đi, chỉ báo chính (sâu, rộng) bị chặn ≤ 5 µm, dài chỉ được ghi. Thay đổi này **không áp dụng ngược** để đổi kết luận "không kết luận" của B1 (đã khóa).

**Dự đoán (ghi trước, có thể sai).** C1 được ủng hộ 30%: (1) đạt ở ít nhất một nhánh 65% (E_id theo kịp E_res khi trường trơn); (2) hụt T_min ≤ 1% ở C1b 55%, ở C1a 25%; (4)–(7) đạt khi (1), (2) đạt 70%. C0 giữ hụt T_min > 1% (90%) và |E_id| > 10 (85%). Ba nhánh khoảng 18 phút.

## 16. Kết quả S2.13 (17:15–17:20, 249 s, mã `gates.energy_identity`, `energy.positivity_term`, `smoke/tier2e.py`): **ĐẠT** cả sáu mục; ghi lúc 17:21

`legacy/ckpt` không đổi.
(a) E_id: B3 **−5.6** mJ (cần −5.6 ± 2; lùi vào 8 µm −4.5); M2 **−25.0** mJ (cần −25.0 ± 2; 8 µm −23.8). (b) trường không có điểm gãy: |E_res − E_id| ≤ 3 mJ: B3 −7.4 / −5.6; M2 −23.5 / −25.0. (c) **M2 cùng trọng số, cắt:** E_res +1.3 mJ (qua theo điểm) nhưng E_id −14.7 mJ, nên cổng (a) mới **TRƯỢT** (a_res đạt, a_id trượt): cổng mới phát hiện nguồn ẩn mà cổng cũ bỏ sót. (d) bốn checkpoint cũ theo cổng (a) mới: B3 đạt; B1k (E_res +100.2, E_id +96.5), B1kc (+31.4, +37.8), B1f1.4 (+15.3, +18.7) trượt, cùng chiều và cùng cỡ với E_res cũ. (e) L_P(M2) = 6.95e-6 (8 mẫu 2^15) so với 6.89e-6 độc lập (tỉ lệ 1.01), gradient hữu hạn, λ = 0 trả [L_PDE, L_top] giống `make_data` (lệch 0) và hai số hạng phụ bằng 0. (f) hồi quy: tầng 2 (S2.1–S2.4) và S2.12 chạy lại **đạt** (ma thoát 0) với mã mới (`light_gate` nay có E_id; `runner.py` đo E_id sau mỗi khối và ghi vào `run.json`). Chấm dự đoán: đạt cả sáu 65%: **đúng**. L_P(M2) = 7e-6 khớp ước lượng 1e-5 của mục 15 (λ_P = 100 cho 7e-4, λ_P = 1000 cho 7e-3).

Chạy C1 ngay theo mục 15.

## 17. Kết quả C1 (17:21–17:34, 13 phút GPU, `experiments/c1_positivity.py`, `runs/C1/report.json`): **C1 ĐƯỢC ỦNG HỘ** theo quy tắc khóa; ghi lúc 17:35

`legacy/ckpt` không đổi. Điểm khởi đầu = M2 (tái lập chính xác: E_res −23.5, E_id −25.0 mJ, hụt T_min 1.48%, bể chảy 525.5 / 166.7 / 98.2 µm). Adam 1,500 bước lr 1e-4; mọi số đo trên điểm độc lập.

| | Bước 0 (M2) | C0 (đối chứng) | C1a (λ_P = 100) | **C1b (λ_P = 1000)** |
|---|---|---|---|---|
| E_res `probe` (mJ) | −23.5 | −25.5 | −0.1 | **+1.1** |
| E_id (mJ) | −25.0 | −26.3 | −2.2 | **−1.1** |
| T_min (K), hụt biên độ | 220.2 K, 1.48% | 221.2 K, 1.46% | 234.7 K, **1.20%** | **251.5 K, 0.88%** |
| tỉ lệ (tích + bay hơi)/nạp | 0.756 | 0.754 | 0.824 | 0.812 |
| gate a / b / c | trượt / trượt / đạt | trượt / trượt / đạt | đạt / **trượt** / đạt | **đạt / đạt / đạt** |
| E_res theo vật liệu: rắn / phân cách / bột (mJ) | −1.6 / −3.0 / −20.4 | −2.6 / −3.3 / −21.0 | +0.3 / −0.9 / −0.8 | +0.9 / −0.8 / −0.3 |
| RMS vùng > 1000 K | 0.0635 | 0.0616 | 0.0660 | 0.0680 |
| `check_physics` [WARN] | 1 | 1 | 1 | **0** |
| chỉ báo 70%: dài / rộng / sâu (µm) | 525.5 / 166.7 / 98.2 | 533.3 / 166.7 / 98.2 | 531.6 / 168.0 / 98.8 | 522.9 / 168.6 / 99.1 |
| tốc độ nguội đáy lớp bột (K/s); T_pk (K) | 1.134e6; 2,962 | 1.145e6; 2,961 | 1.108e6; 2,975 | 1.153e6; 2,980 |

Tiêu chí (mục 15): **C1b đạt cả bảy** (1: |E_res| 1.1, |E_id| 1.1 ≤ 10; 2: hụt T_min 0.88% ≤ 1%; 3: tỉ lệ 0.812 ≤ 1; 4: |E_c| ≤ 0.9 ≤ 7; 5: RMS 0.0680 ≤ 0.0762; 6: sâu +0.9, rộng +1.9 µm ≤ 5; 7: 0 cảnh báo). C1a đạt (1), (3)–(6) nhưng trượt (2) (hụt 1.20%) và (7). C0 trượt (1), (2), (4), (7). **C1 được ủng hộ**: có nhánh đạt tất cả và C0 trượt (1) và (2); λ* = C1b (λ_E = 1, λ_P = 1000).

Chấm dự đoán (mục 15): C1 được ủng hộ 30% → **xảy ra**; (1) đạt ở ít nhất một nhánh 65% → đúng (cả hai); hụt T_min ≤ 1% ở C1b 55% → đúng, ở C1a 25% → đúng (không); C0 giữ hụt > 1% (90%) và |E_id| > 10 (85%) → đúng cả hai. Thời gian dự báo 18 phút, thực tế 13.

**Hạn chế phải ghi cùng kết quả (không cắt bớt):**
1. **Dao động giữa các điểm kiểm.** Cổng nhẹ ở bước 500 / 1000 / 1500 cho C1b: E_res +5.0 / −4.5 / +1.1 mJ, E_id +0.9 / −8.5 / −3.2 mJ; C1a: E_res +4.5 / −3.7 / −0.1, E_id −0.1 / −7.7 / −4.4. Biên độ ±5–9 mJ gần ngưỡng 10 mJ; kết quả cuối nằm ở phía thuận lợi và **có thể là may mắn một phần** (một lần chạy, một điểm cuối). Một lượt chính thức cần quy tắc dừng yêu cầu **hai điểm kiểm liên tiếp** đạt (như M2).
2. **Biên của gate (b) hẹp:** 0.88% so với ngưỡng 1% (cải thiện đều: 1.05 → 0.98 → 0.89 qua ba điểm kiểm); λ_P = 100 chưa đủ (1.20%).
3. **λ_P chọn từ hai giá trị** (100 và 1000) bằng chính các kết quả này (quy tắc mục 15 cho phép), nên chưa độc lập; chưa kiểm khả năng tổng quát hóa sang lượt đầy đủ từ B3.
4. Đây là tinh chỉnh 1,500 bước từ trọng số M2, **không** phải lượt M2 đầy đủ từ B3: M2 và M2′ vẫn TRƯỢT (không đổi).
5. Số hạng phụ lấy **mẫu ngẫu nhiên mới ở mỗi bước** nên mục tiêu không tất định; **không dùng được với L-BFGS** (tìm kiếm đường cần hàm mục tiêu cố định). Lượt chính thức vì vậy chỉ có Adam, hoặc phải dùng mẫu cố định cho L-BFGS (chưa thiết kế, chưa kiểm).
6. RMS vùng nóng tăng nhẹ so với C0 (0.0680 so với 0.0616, +10%): chi phí nhỏ của hai số hạng phụ.

**Số đo bể chảy (chưa phải chỉ báo, vì chưa có lượt chính thức qua cổng).** Độ sâu 70% của mọi biến thể trên nêm: M2 98.2, M2′ 97.9, B1 λ = 1 98.0, B1 λ = 10 98.4, C0 98.2, C1a 98.8, C1b 99.1 µm, tức khoảng dao động **1.2 µm** (ΔD so với B3, 90.8 µm: +7.1 đến +8.3 µm); độ rộng 166–169 µm (khoảng 3 µm). Chiều dài 512–535 µm (khoảng 23 µm) và tốc độ nguội 1.06–1.27e6 K/s vẫn nhạy với cách huấn luyện. Tỉ lệ (tích + bay hơi)/nạp của C1b là 0.812, gần B3 vệt thường (0.805), còn M2 là 0.756.

## 18. Đăng ký 03d (17:37, trước khi viết mã và chạy): M2″ — lượt chính thức từ B3 với số hạng cân bằng năng lượng và phạt dương

Người dùng đồng ý (sau mục 17): viết đăng ký, chạy M2″, rồi chuyển sang hướng 4. M2 và M2′ vẫn TRƯỢT (không đổi); M2″ là một lượt **mới** từ B3, không phải tinh chỉnh từ M2.

**Cấu hình.** Bậc `B3W0` (nêm y_a = 0, **không cắt**, vật lý B3), khởi động từ `B3_84d0289e` (chỉ đọc), 4 seed Sobol cố định (K = 0), seed 42, mạng 6 × 24, float32. **Chỉ Adam** (L-BFGS không dùng được với số hạng phụ ngẫu nhiên mỗi bước, mục 17 hạn chế 5), lr 1e-4 toàn lô, `compile` một lần. Loss = L_PDE + L_top + 1·L_E + 1000·L_P (λ_E = 1, λ_P = 1000 = λ* của C1, mục 17; định nghĩa L_E, L_P như mục 11 và 15).

**Điểm kiểm và quy tắc dừng.** Cứ **500 bước** đo **cổng đầy đủ** (`probe` 2^21 điểm, E_id, T_min; cổng (a) = |E_res| và |E_id| ≤ 10 mJ; (b) hụt T_min ≤ 1%; (c) tỉ lệ ≤ 1). **Dừng** khi **hai điểm kiểm liên tiếp đều đạt cả ba cổng** và đã qua bước **2,000** (không dừng sớm hơn, để có ít nhất lượng Adam tương đương C1 và M2), hoặc hết **6,000 bước**, hoặc trần giờ **45 phút** (không bắt đầu đoạn mới nếu đã quá), hoặc loss không hữu hạn. Chạy dưới `supervise` (thử lại một lần nếu sập; Adam khôi phục từ điểm kiểm cuối, trạng thái tối ưu hóa được khởi tạo lại, ghi rõ trong log).

**M2″ ĐẠT khi (tất cả):** (a) mọi loss và số đo hữu hạn, không sập quá một lần; (b) **dừng vì hai điểm kiểm liên tiếp đạt cổng đầy đủ**, và cổng đầy đủ cuối (đo lại riêng sau khi dừng, trên trọng số cuối, `probe` seed 101) đạt (a), (b), (c); (c) `check_physics` 0 [WARN] trên trọng số cuối; (d) ≤ 75 phút. **Trượt thì dừng, báo cáo, không chạy lại.**

**Chỉ ghi, không phải tiêu chí:** đường E_res, E_id, hụt T_min theo bước; E_res theo vật liệu (3 lớp); RMS residual vùng > 1000 K (B3 gốc 0.0398, M2 0.0635, C1b 0.0680); bể chảy và chỉ báo ở 50/70/90% (dài, rộng W₊ + W₋, sâu, độ xuyên) và chu trình nhiệt tại đáy lớp bột; so với B3 (M1), M2, C1b. Chiều dài bể chảy chỉ ghi.

**Mốc test S2.14 (chạy trước M2″; trượt thì dừng, báo cáo):** thay đổi mã: `runner.py` có chế độ Adam với số hạng phụ (`--energy-lam`, `--pos-lam`), cổng đầy đủ ở điểm kiểm Adam (`--monitor-every`, `--stop-gates`, `--min-steps`), khôi phục giữa Adam, `--max-iters 0` bỏ L-BFGS.
| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | Logic dừng: `should_stop` có bước tối thiểu trên chuỗi giả | [đ, đ] ở bước < 2,000 không dừng; [t, đ, đ] ở bước ≥ 2,000 dừng; [đ, t, đ] không dừng |
| (b) | Khói: `runner.py run B3W0 --adam-iters 80 --monitor-every 20 --stop-gates 2 --min-steps 40 --max-iters 0 --energy-lam 1 --pos-lam 1000` (thư mục tạm) | ra `run.json` DONE; mỗi điểm kiểm có cổng đầy đủ kèm E_id; tag khác `B3W0_fff2a3d4` và khác tag của M2; dtype giữ float32 sau mỗi lần đo cổng (80 bước chạy hết, loss hữu hạn); không L-BFGS |
| (c) | Khôi phục: giết cứng sau điểm kiểm đầu, `--resume` | `run.json` DONE, `resumed_from_block` = 0 nhưng bước Adam tiếp từ `adam_done` ≥ 20, tổng bước = 80 (không chạy lại từ 0) |
| (d) | Hồi quy: tầng 2 (S2.1–S2.4), S2.5–S2.9 (tầng 2b) chạy lại | đạt |
Trần GPU 10 phút (dừng ở 15). Dự đoán S2.14 đạt cả bốn lượt đầu 70%.

**Lệnh M2″:** `python runner.py supervise B3W0 --out runs/M2pp --adam-iters 6000 --adam-lr 1e-4 --chunk 500 --max-iters 0 --monitor-every 500 --stop-gates 2 --min-steps 2000 --time-cap-s 2700 --energy-lam 1 --pos-lam 1000`.

**Dự đoán (ghi trước, có thể sai).** M2″ ĐẠT 50%. Lý do giữ không chắc: C1b qua cổng từ M2 (đã huấn luyện 5,500 bước) chứ chưa từ B3; điểm kiểm dao động ±5–9 mJ nên "hai điểm kiểm liên tiếp đạt" là điều kiện chặt (xác suất dừng theo cổng trong 6,000 bước khoảng 60%); `check_physics` có thể có cảnh báo ở các phép 5–8 (viết cho vệt đối xứng; M2 và C1b không có). Nếu trượt, nghi phạm: gate (b) (T_min sát 1%) 40%, E_id/E_res dao động 40%, `check_physics` 20%. Thời gian: 12–24 phút.

**Bước sau M2″.** ĐẠT: ghi kết quả; **hướng 4** theo yêu cầu người dùng: viết đăng ký 04 (độ nhạy của chỉ báo với hấp thụ α_s trong dải 0.30–0.60 của Moser 2019; mỗi giá trị cần một cặp vệt thường + nêm qua cổng; thiết kế ở đăng ký 04). M3a/M3b/M4 của đăng ký 02 để lại, hỏi người dùng. TRƯỢT: dừng, báo cáo và thảo luận trước khi sang hướng 4.

## 19. Kết quả S2.14 (17:39–17:44; mã `runner.py` chế độ Adam, `smoke/tier2f.py`): **ĐẠT, kèm một sự cố sập native**; ghi lúc 17:44

`legacy/ckpt` không đổi. (a) `should_stop` có bước tối thiểu: 5/5 chuỗi giả đúng. (b) khói Adam 80 bước (kiểm mỗi 20, `--stop-gates 2 --min-steps 40`, λ_E = 1, λ_P = 1000, thư mục tạm): 19 s, thoát 0, 3 điểm kiểm đều có E_id, dừng `gates_passed`, tag `B3W0_fe47ed79` (khác `B3W0_fff2a3d4` và M2 `B3W0_e90da666`), loss hữu hạn, 0 vòng L-BFGS, cổng đầy đủ cuối đạt; (c) khôi phục: giết sau điểm kiểm đầu (adam_done = 20), `--resume` thoát 0, tổng 80 bước Adam, 4 điểm kiểm, log có "khôi phục từ bước 20" (dtype giữ đúng giữa các lần đo cổng). (d) hồi quy: tầng 2b (S2.5–S2.9) **đạt**; tầng 2 (S2.1–S2.4) ở **lần đầu sập** với mã thoát `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN) trong tiến trình kiểm thử chính ngay sau khi S2.1 đạt (lúc vào S2.2), **không** trong `runner.py`; chạy lại một lần: **đạt cả bốn trong 65 s**. Đây là sập native đơn lẻ không tái lập, thuộc loại đã ghi ở kế hoạch gốc (mục 8, "sập native không phụ thuộc thư viện; xử lý bằng resume và thử lại một lần"); ghi lại vì tầng 2 đã chạy bốn lần trước không sập, lần này sập một lần trong năm. Quy tắc 1 của đăng ký 01 (trượt thì dừng) áp dụng cho trượt tiêu chí, không cho sập hạ tầng đã có cơ chế thử lại; tôi coi (d) đạt sau thử lại một lần, **như đã nêu**. Hệ quả cho M2″: `supervise` (thử lại một lần bằng `--resume`, nay khôi phục được ở giữa Adam) bao phủ trường hợp này.
Chấm dự đoán (mục 18): S2.14 đạt cả bốn lượt đầu (70%) → **sai một phần** (một sự cố sập hạ tầng ở (d), tiêu chí (a)–(c) đạt ngay lượt đầu). Chạy M2″ ngay theo mục 18.
