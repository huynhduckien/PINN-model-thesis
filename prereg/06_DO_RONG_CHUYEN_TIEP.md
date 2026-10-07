# Đăng ký trước 06 — độ nhạy của chỉ báo cách nhiệt theo độ rộng chuyển tiếp của hệ số dẫn nhiệt

Ngày: 2026-10-07 12:20 (đồng hồ; bản đầu ghi nhầm 12:25, sửa theo đồng hồ). Người dùng đồng ý: (1) khóa tiêu chí và chạy mốc test trước; (2) chạy 15 và 45 µm cho vệt thường và nêm; (3) cập nhật bảng chỉ báo và mục phạm vi sau khi có kết quả; **nếu 15 µm trượt cổng thì chỉ báo hai điểm (không có điểm dự phòng 22 µm)**; **quét một hệ số chung cho hai độ rộng mép, giữ nguyên w_x**. **Commit file này trước khi viết mã và trước khi chạy bất kỳ mốc nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và phạm vi

Mô hình bột cách nhiệt (`05` mục 10, 13, 17) có một giả định chưa kiểm: độ rộng 10–90% của vùng chuyển tiếp hệ số dẫn nhiệt = **30 µm** (giả định cỡ hạt D50). Nó vừa là yêu cầu số (mép sắc 8 µm không qua cổng, `05` mục 8) vừa là giả định vật lý. Bột cách nhiệt đã đổi ΔD từ +9.0 lên +18.3 µm, nên kết quả chính có thể phụ thuộc nó.

**Câu hỏi:** khi độ rộng chuyển tiếp đổi từ 30 µm sang **15 µm** và **45 µm**, (i) PINN có qua cổng trung thực và mặt nạ có hội tụ không; (ii) các Δ (nêm − vệt thường) của độ sâu/độ xuyên, T_pk, thời gian lưu, tỉ số tốc độ nguội đổi bao nhiêu so với 30 µm (+18.3 µm; +160 K; +222 µs; 0.543)?

Đây là **nghiên cứu độ nhạy**, không phải hiệu chỉnh: không có dữ liệu thực nghiệm, **không** nói độ rộng nào đúng. Không làm M3 (lặp ba biến thể) cho các độ rộng mới: mỗi (độ rộng, hình học) chỉ **một lượt**.

## 2. Định nghĩa khóa

Độ rộng 10–90% W ∈ {15, 45} µm (30 µm là tham chiếu đã có). **Một hệ số chung** đổi cả hai độ rộng mép theo đúng quan hệ của `05` mục 10:
- mặt phân cách lớp bột trong κ_s: tanh trên thang a_I = W/2.197 µm (W = 30 → 13.65 µm);
- mép bao bể chảy: sigmoid theo khoảng cách có dấu, w_e = W/4.394 µm (W = 30 → 6.83 µm).

**Giữ nguyên (không quét):** w_x = 8 µm (mép mũi bể theo hướng quét), r = 0.02346, x_nose = 47 µm, nhiệt dung bột (bản đồ vật liệu cũ, không đổi), bảng bao vòng 0 (`runs/ENV/B3_iter0.pt`, `B3W0_iter0.pt`: bảng T_max không phụ thuộc W; chỉ cách hiểu e đổi), công thức tối ưu của `05` mục 13 (Adam, λ_E = 1, λ_P = 1000, lr 1e-4, khởi động từ B3 và M2″, `supervise`), tiêu chí ĐẠT và hội tụ mặt nạ. **Một thay đổi duy nhất so với `05` mục 13: W.** Điều này nghĩa là độ nhạy kiểm được là độ nhạy theo độ dày vùng chuyển tiếp của **mặt phân cách lớp bột và mép bao**; **không** phủ độ mềm của mép mũi bể (w_x), không phủ cỡ hạt như một tham số vật lý khác (k bột, nhiệt dung).

Bậc mới: `B3s15`, `B3W0s15`, `B3s45`, `B3W0s45` (thuộc `config.RUNS`, `soft = dict(a_iface_um = W/2.197, w_e_um = W/4.394)`); `B3s`, `B3W0s` (30 µm) **không được đổi** (S2.20(a) kiểm).

## 3. Mốc test S2.20 (chạy trước; trượt mục nào thì dừng, báo cáo, không chạy lượt đầy đủ)

| ID | Kiểm | Tiêu chí |
|---|---|---|
| (a) | Cấu hình: `RUNS["B3s"]`, `RUNS["B3W0s"]` giữ đúng `soft` cũ (30.0/2.197, 30.0/4.394) và `_SOFT` không đổi; bốn bậc mới có `soft` = (W/2.197, W/4.394) và mọi khóa khác (init, geometry, physics, insulated) bằng bậc 30 µm tương ứng; `scales()` của bậc mới trùng `scales()` của bậc 30 µm (trừ tên) | đúng |
| (b) | Độ rộng chuyển tiếp thực đo (như S2.18(a), vệt thường, W = 15 và 45): mặt phân cách lớp bột và mép bao | mặt phân cách: 10–90% = W ± 1.5 µm, tâm 180 ± 1 µm; mép bao: 10–90% = W ± 2.5 µm, tâm mép +y trong ± 1 µm so với trung điểm pixel nóng cuối và pixel lạnh đầu (**ngưỡng tuyệt đối như S2.18, không nới**) |
| (c) | Nhất quán bao (như S2.18(b)) cho W = 15 và 45 | độ sâu và nửa rộng của vùng e ≥ 0.5 lệch ≤ 2 µm so với `envelope_stats` |
| (d) | Đạo hàm và divergence (như S2.17(b), (c), nêm, float64, 2^12 điểm) với cấu hình mềm W = 15 và 45 | phân vị 99%: ≤ 1e-4 (∇κ_s) và ≤ 1e-5 (div) |
| (e) | κ_s tại điểm (vệt thường, sau mũi, ngoài bao, như S2.18(e)), W = 15 và 45: z = 180, 210, 150 µm, và mép bao +y tại mặt trên | khớp công thức giải tích (độc lập, tính trong test) ± 0.02 (mặt phân cách), ± 0.05 (mép bao) |
| (f) | Canary tối ưu hóa: cả bốn bậc mới, **500 bước** Adam theo công thức `05` mục 13 (`runner.py supervise`, `--stop-gates 0`, kiểm 250, từ B3/M2″, bao vòng 0), thư mục tạm | mọi `run.json` DONE; tag bắt đầu bằng tên bậc, bốn tag khác nhau và khác tag của `B3s_d2f9f1db`, `B3W0s_ba6419f6`; loss hữu hạn; ≤ 0.25 s/bước; `legacy/ckpt` không đổi. **Chỉ ghi, không phải điều kiện:** tỉ số loss bước 500 / bước 0, E_res, E_id, hụt T_min (500 bước không đủ phân biệt "không huấn luyện được"; cổng trung thực của lượt đầy đủ là phán quyết) |
| (g) | Hồi quy: tầng 2 (S2.1–S2.4), S2.13, S2.17 và S2.18 (cấu hình 30 µm không đổi) chạy lại | đạt |

Trần GPU mốc test 30 phút (dừng ở 45). Dự đoán: (a)–(e) đạt cả ngay lượt đầu 65% (đã có hai lỗi đặc tả điểm kiểm ở S2.17, S2.18; điểm (b) ở W = 15 nhạy với lưới 2 µm); (f) đạt 85%; (g) đạt 85%.

## 4. Lượt đầy đủ G (chạy nếu S2.20 đạt cả bảy mục; người dùng đã yêu cầu chạy 15 và 45)

`python experiments/f_insul.py run --soft --width W` cho W = **45 rồi 15** (độ rộng dễ hơn trước, để giữ kết quả chắc chắn trước khi chạm điểm rủi ro). Thư mục `runs/Fw45/`, `runs/Fw15/`. Mỗi độ rộng: vệt thường trước, rồi nêm; mỗi lần huấn luyện ĐẠT theo `05` mục 4 (dừng vì hai điểm kiểm liên tiếp đạt sau 2,000 bước, cổng đầy đủ cuối đạt, `check_physics` 0 [WARN], ≤ 75 phút); vòng lặp mặt nạ tối đa 3 lần huấn luyện lại; trần mỗi độ rộng 120 phút (dừng ở 180). **Vệt thường trượt cổng → dừng độ rộng đó, không chạy nêm. Trượt thì ghi, không chạy lại, không đổi ngưỡng cổng.**

**Cặp hợp lệ cho độ rộng W** = vệt thường và nêm đều ĐẠT mọi lần huấn luyện, mặt nạ nêm hội tụ (mặt nạ vệt thường hội tụ hoặc vệt thường dừng ở vòng 0 như `05` mục 13 khi hội tụ ngay). Độ rộng không có cặp hợp lệ được báo là **"không huấn luyện được ở độ rộng này"** (không phải số liệu); khi đó báo cáo độ nhạy chỉ bằng các độ rộng còn lại (30 µm và độ rộng hợp lệ kia). Nếu cả hai độ rộng mới đều không hợp lệ: ghi **"chưa kiểm được độ nhạy"** (chỉ có 30 µm).

## 5. Chỉ báo, so sánh và quy tắc kết luận (khóa)

**Chỉ báo** (từ trọng số cuối bằng chính `analysis.net_indicator`, như `05` mục 14): ở 70% quét D, P = max(D − 60 µm, 0), W₊, W₋ (chiều dài chỉ ghi, R1); chu trình nhiệt ở **50%** quét tại (y = 0, đáy lớp bột): T_pk, thời gian lưu trên T_sol, tốc độ nguội. Δ = nêm − vệt thường; tỉ số nguội = nêm/thường.

**Tham chiếu 30 µm:** ΔD = ΔP = +18.31 µm, ΔT_pk = +159.8 K, Δlưu = +222.0 µs, tỉ số nguội 0.543 (lượt tham chiếu, `05` mục 17); nhiễu huấn luyện đo được ở 30 µm (độ lệch lớn nhất giữa ba biến thể): 0.41 µm, 10.9 K, 10.0 µs, 0.041. Cơ sở bột như đặc: ΔD = +9.0 µm.

**Với mỗi độ rộng hợp lệ W và mỗi chỉ báo X ∈ {ΔD, ΔT_pk, Δlưu, tỉ số nguội}**, d_X = |X(W) − X(30)| và phân loại:
- **trong nhiễu**: d_X < 2 × nhiễu_30(X) (0.82 µm; 21.8 K; 20.0 µs; 0.082);
- **nhạy vừa**: 2 × nhiễu_30(X) ≤ d_X ≤ dung sai S-b (3 µm; 15 K; 50 µs; 0.10);
- **nhạy**: d_X > dung sai S-b (dung sai của `05` mục 14, **[ĐỀ XUẤT]** khi đó, không đổi).
Với ΔT_pk: nhiễu 21.8 K > dung sai 15 K của S-b nên lớp "nhạy vừa" rỗng; quy tắc: ΔT_pk **nhạy** khi d > 21.8 K, ngược lại **trong nhiễu**. Nhiễu ở W ≠ 30 µm **chưa đo** (một lượt); việc dùng nhiễu_30 cho W khác là **giả định** và ghi kèm.

**Quy tắc kết luận (khóa, theo thứ tự):**
1. **Hướng:** nếu tại mọi độ rộng hợp lệ ΔD > 0, ΔT_pk > 0, Δlưu > 0 và tỉ số nguội < 1 → "**hướng của hiệu ứng không đổi theo độ rộng**" (ngược lại ghi chỉ báo đổi dấu ở độ rộng nào).
2. **Cách nhiệt tăng hiệu ứng:** nếu ΔD(W) > +9.0 µm tại mọi độ rộng hợp lệ → "hiệu ứng cách nhiệt (so với bột như đặc) giữ ở mọi độ rộng đã kiểm"; ngược lại ghi độ rộng nào ΔD ≤ +9.0.
3. **Độ nhạy từng chỉ báo:** nếu mọi độ rộng hợp lệ ở lớp "trong nhiễu" → "**không nhạy**" (bảng cuối giữ số 30 µm ± nhiễu và ghi "kiểm ở các độ rộng đã chạy"); nếu có "nhạy vừa" hoặc "nhạy" → bảng cuối **báo khoảng** [min, max] qua mọi độ rộng hợp lệ kể cả 30 µm, thay cho một con số ± nhiễu. Đại lượng chỉ ghi: chiều dài, ΔW (không phải chỉ báo, `05` mục 17).
4. Không nội suy hay ngoại suy ngoài {đã chạy}; hai hoặc ba điểm **không** xác định dạng hàm và không cho "độ rộng thật".

## 6. Đầu ra

(i) `runs/Fw15/`, `runs/Fw45/` (nhật ký, `run.json`, trọng số), `runs/Fw/summary.json` (do `experiments/g_width.py summarize`), (ii) bảng: với mỗi độ rộng (15, 30, 45 µm): trạng thái hợp lệ, D, T_pk, lưu, nguội mỗi hình học, các Δ, lớp phân loại; (iii) cập nhật bảng chỉ báo cuối cùng và mục phạm vi của `05` (mục 18 mục 6, 7) theo quy tắc ở mục 5 (người dùng yêu cầu việc này), ghi trong mục mới của **file này**; không sửa mục cũ của `05`, chỉ thêm mục tham chiếu.

## 7. Dự đoán (ghi trước, có thể sai)

- Vệt thường 45 µm qua cổng 70%; nêm 45 µm qua cổng, nếu vệt thường qua: 70%. Vệt thường 15 µm qua cổng 45% (mép sắc 8 µm trượt, 30 µm đạt; E_id biên mỏng −8.8 mJ ở 30 µm); nêm 15 µm qua cổng, nếu vệt thường qua: 55%. Mặt nạ hội tụ ≤ 3 lần huấn luyện lại, nếu qua cổng: 65%.
- Nếu hợp lệ: hướng không đổi 90%; ΔD > +9.0 µm ở mọi độ rộng hợp lệ 80%; ΔD thu hẹp khi W tăng (mặt phân cách mờ hơn làm cách nhiệt kém hiệu quả hơn): 65%; ΔD(45) trong ± 3 µm của 18.3: 45%; ΔD(15) trong ± 3 µm: 35%; ΔT_pk trong nhiễu ở W = 45: 40%; thời gian lưu "nhạy" (> 50 µs lệch) ở ít nhất một độ rộng: 45%.
- Tổng: kết quả cuối "ΔD báo khoảng, không phải một con số" (có "nhạy vừa" hoặc "nhạy" ở ΔD): 70%.
- GPU: mốc test 20–35 phút; lượt đầy đủ 20–45 phút mỗi độ rộng.

## 8. Giới hạn (khóa)

Một lượt mỗi (độ rộng, hình học): nhiễu ở W ≠ 30 µm là **giả định** lấy từ 30 µm; chỉ ba điểm W (hai điểm nếu một độ rộng trượt); quét một hệ số chung cho hai mép, **w_x = 8 µm không quét**; mọi giả định khác của `05` mục 18 (mặt nạ tĩnh theo bao cực đại, k bột Moser ngoại suy, nhiệt dung bột không đổi, đáy giữ nhiệt độ môi trường, một điều kiện 300 W/0.8 m/s, 45°, α_s = 0.35, chưa có dữ liệu ngoài) giữ nguyên và **chưa kiểm**; kết quả là độ nhạy của **mô hình PINN này** theo W, không phải độ nhạy của vật lý thật. Trượt cổng ở một độ rộng là kết quả về công thức tối ưu ở độ rộng đó, không phải về vật lý.

## 9. Sửa đổi 06a (12:23, sau khi viết mã, trước khi chạy mốc test GPU): thêm mục (h) và làm rõ (a), (f)

1. **Thêm (h) vào S2.20 (chỉ làm mốc test chặt hơn):** bộ đánh giá độ nhạy của `experiments/g_width.py` (quy tắc mục 5) chạy trên **8 trường hợp giả** (trong nhiễu ở cả hai độ rộng; ΔD lệch 2 µm → nhạy vừa và báo khoảng; ΔD lệch 5 µm → nhạy; ΔT_pk 18 K trong nhiễu / 25 K nhạy, không có lớp "vừa"; một độ rộng không hợp lệ; cả hai không hợp lệ → "chưa kiểm được"; ΔD ≤ +9.0 µm ở một độ rộng; tỉ số nguội ≥ 1 → hướng đổi). Tiêu chí: cả 8 cho đúng kết luận. Thứ tự chạy: a, b, c, d, e, h, f, g.
2. **(a):** so sánh `scales()` bỏ khóa `run` (tên bậc), vì chỉ khóa đó khác theo thiết kế.
3. **(f):** để ghi loss khởi điểm (bước 0), dùng `tier2i.step0_loss` cho bốn bậc (chỉ ghi); canary chạy `supervise` đúng như mục 3; tốc độ lấy từ `adam.seconds / 500` (cùng cách đo như lượt F: 0.17 s/bước).
(a) và (h) là kiểm thuần Python (không GPU) nên đã chạy thử trước khi ghi mục này; cả hai đạt, không ảnh hưởng đến bất kỳ phán quyết nào.

## 10. Kết quả lượt đầu của S2.20 và sửa đổi 06b (13:21): lỗi của chính khung kiểm ở (g)

Lượt đầu (12:22–13:19, `smoke/tier2l.py`, log `runs/tier2l/main_run1.log`, `g_run1.log`; mã `7b4a859`): **(a), (b), (c), (d), (e), (h), (f) đạt; (g) trượt**; tôi dừng lượt chạy ở 13:19 (đã trượt, phần còn lại không đổi phán quyết).
- (a) cấu hình đúng. (b) độ rộng chuyển tiếp: W = 15: mặt phân cách 15.00 µm (tâm 180.00), mép bao 14.73 µm (tâm mép +y 85.00); W = 45: 45.02 µm (tâm 180.00), mép bao 44.92 µm (tâm 85.00). (c) bao e ≥ 0.5: sâu 90.0, nửa rộng 83.0 µm (khớp `envelope_stats`). (d) ∇κ_s: 2.1e-7 (W = 15), 8.9e-9 (45); div: 2.6e-7, 2.1e-7. (e) κ_s khớp công thức tới 4 chữ số (W = 15: 0.5117 / 0.0236 / 0.9999 / 0.5117; W = 45: 0.5118 / 0.0731 / 0.9505 / 0.5131). (h) bộ đánh giá: 8/8.
- (f) canary 500 bước (369 s, 0.173–0.175 s/bước): DONE 4/4, bốn tag khác nhau và khác tag tham chiếu, loss hữu hạn. Chỉ ghi, [PDE, top] bước 0 → 500: B3s15 [0.0705, 1e-4] → [0.0262, 0.0014]; B3W0s15 [0.0911, 1e-4] → [0.0385, 0.0010]; B3s45 [0.0544, 1e-4] → [0.0116, 4e-4]; B3W0s45 [0.0668, 1e-4] → [0.0207, 5e-4]. E_res / E_id (mJ) ở bước 500: 15 µm: +3.0 / −12.9 (thường), +6.9 / −6.8 (nêm); 45 µm: +6.6 / +1.0, +6.3 / −2.1. Hụt T_min 0.63–0.78%. (Ghi nhận: 500 bước chưa phân biệt cổng; E_id của B3s15 đã ngoài ngưỡng 10 mJ ở bước 500, chưa phải phán quyết.)
- **(g) TRƯỢT:** S2.17(h) (một mục của tầng hồi quy, chạy lại qua `tier2i.py`) báo **1.86 và 1.83 s/bước** so với ngưỡng ≤ 0.25 (tổng loss 0.441 và 0.455 lần, E_id −22.1 / −10.0 mJ khớp lượt gốc). Tại lúc kiểm, `nvidia-smi` cho **15,994 / 16,376 MiB đang dùng** và có tám tiến trình tính toán: `tier2l.py` cha (giữ bộ nhớ GPU sau (b)–(e), không giải phóng), `tier2j.py` cha, `tier2i.py` (lồng), và trình huấn luyện con. VRAM đầy làm tốc độ chậm khoảng 10 lần. Đây là **lỗi của khung kiểm của tôi** (tiến trình cha giữ bộ nhớ trong khi chạy hồi quy lồng nhau), cùng loại với lỗi bộ nhớ ở `01f`, không phải lỗi mã hay lỗi tối ưu hóa; **số loss và E của mục đó khớp lượt gốc**.

**Sửa (chỉ khung chạy (g); ngưỡng 0.25 s/bước và mọi tiêu chí khác không đổi):** (g) chạy như một lượt **riêng** `python smoke/tier2l.py g` sau khi tiến trình chính đã thoát (tiến trình cha không giữ bộ nhớ GPU), và `nvidia-smi` phải cho bộ nhớ trống < 1 GiB trước khi bắt đầu. Các mục (a)–(f), (h) **không chạy lại**: mã của chúng không đổi giữa hai lượt (chỉ thêm hàm `regression` và chế độ `g`), kết quả ở trên đứng. Lượt đầy đủ G chỉ bắt đầu khi (g) đạt.
