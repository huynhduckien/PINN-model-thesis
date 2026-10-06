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
