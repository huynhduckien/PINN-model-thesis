# Kế hoạch quy trình code và kế hoạch chạy — PINN overhang không support (LPBF)

Ngày: 2026-10-06. **Bản nháp để bạn duyệt; chưa chạy gì, chưa viết code.** Mọi con số thời gian lấy từ số đo đã ghi trong dự án; chỗ nào là ước lượng của tôi thì ghi "ước lượng". Mọi ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất, bạn quyết.

## 1. Mục tiêu và ràng buộc đã chốt

- Một điều kiện cố định (không tham số hóa): 300 W, 0.8 m/s, r_b 40 µm, lớp bột 60 µm, IN718, một đường quét.
- PINN thuần vật lý, **không FD**; kiểm chứng bằng nghiệm giải tích, MMS, cân bằng năng lượng, cổng vật lý.
- Đầu ra: chỉ báo nhiệt **tương đối** overhang so với vệt thường (độ xuyên dưới đáy lớp bột, tích tụ nhiệt, tốc độ nguội); không dự đoán dross hay lún.
- Hiệu chỉnh hấp thụ theo các điều kiện đã chốt (chia tập, tham số bị chặn, gọi là bán thực nghiệm).
- Thư mục mới, tách khỏi `coding`, có `git init`; mỗi bản đăng ký trước được commit **trước** khi chạy.
- Không chạy bất kỳ lượt huấn luyện nào trước khi qua các mốc test (yêu cầu của bạn).

## 2. Điểm quyết định D0 (phải có trước khi chạy gì)

Điểm vận hành "300 W và không bay hơi" không nhất quán về vật lý: thông lượng bay hơi theo công thức Fang bằng 0.5% cường độ hấp thụ ở T_vap, **11% ở 4,000 K và 92% ở 5,000 K**, trong khi mô hình không bay hơi cho T_peak khoảng 8,200 K. Hai hướng:

| | **(B) giữ 300 W, khôi phục bay hơi** (đề xuất) | **(A) hạ công suất, bỏ bay hơi** |
|---|---|---|
| Nền vệt thường | `B3_84d0289e` đã đạt 9/9 (52 phút), **không cần chạy S1** | phải hoàn tất S1: `B1f2` (đã có 500 vòng L-BFGS), `B1f2.8` (ước lượng 1.2 giờ), rồi thêm bước đổi P |
| Nhất quán vật lý | B3: sâu/nửa rộng = 90/83 = 1.08, trong chế độ dẫn nhiệt (≲ 1.2) | cần chọn P ≈ 100–130 W (ước lượng thô từ nghiệm R0 tuyến tính), chưa kiểm |
| Rủi ro | độ cứng ở mặt trên đã gặp ở B2, nhưng đã vượt qua | tổng thời gian chưa biết |
| Mâu thuẫn quyết định cũ | trái quyết định "không bay hơi" | không |

Phần còn lại của kế hoạch viết cho (B); khác biệt của (A) ghi ở mục 7.

## 3. Quy trình code (thư mục mới `lpbfpinn/`, đề xuất `d:\KIEN 2026\paper\Luan van tien si\lpbfpinn\`)

Giữ DeepXDE làm nền (đã kiểm: `train_next_batch` ~0 s, 0.13 s/bước, tương đương bài PyTorch thuần của Sharma). Không viết lại vật lý; chỉ thêm các phần mỏng:

```
lpbfpinn/
  legacy/lpbf_pinn.py      bản đóng băng, chép nguyên (63 hàm, đã có nêm)
  legacy/ckpt/             chép A1s, B1, B2, B3 (không chép cái khác)
  runner.py                chạy một giai đoạn: Adam, L-BFGS float64 theo khối 500, lưu checkpoint từng khối,
                           resume, cổng dừng vật lý, trần giờ, thử lại một lần nếu tiến trình sập (mã 0xC0000409)
  gates.py                 cùng một bộ cổng cho: dừng sớm, chạy khói và phán quyết cuối
                           (nguồn giả E_res, T_min, tích/nạp), có bản nhẹ (2^18 điểm) và bản đầy đủ (probe 2^21)
  indicator.py             chỉ báo overhang đọc từ trường nhiệt (độ xuyên dưới z = 180 µm, tích tụ nhiệt, tốc độ nguội)
  smoke/                   các mốc test (mục 4)
  prereg/                  đăng ký trước, mỗi file một commit trước khi chạy
  runs/<id>/               run.json (hash cấu hình, ngân sách, thời gian, cổng), log, checkpoint
```

Quy tắc vận hành (rút từ sự cố hôm nay, mỗi quy tắc có lý do đo được):
1. **Khóa GPU:** một tệp khóa; tiến trình thứ hai từ chối chạy (hôm nay một lượt đo song song làm Adam của `B1f2` chậm 3 lần).
2. **Giám sát sống:** mỗi 3 phút kiểm tiến trình, mtime của log và mức dùng GPU; không chỉ đọc log (hôm nay chuỗi sập lúc 11:35, tôi chỉ thấy lúc 12:18).
3. **Checkpoint mỗi khối + `--resume`** (đã có nguyên mẫu trong `overhang_s1h.py`).
4. **Trần giờ và quy tắc dừng khóa trong đăng ký**, kiểm tự động.
5. Mỗi lượt kiểm `date` và PID trước khi can thiệp tiến trình nền.

## 4. Các mốc test (chạy theo tầng; trượt tầng nào thì dừng, không sang tầng sau)

**Tầng 0 — tĩnh, không huấn luyện (ước lượng ≤ 15 phút)**

| ID | Kiểm | Tiêu chí | Bắt lỗi gì |
|---|---|---|---|
| S0.1 | Môi trường: phiên bản torch/deepxde, GPU rảnh | không có tiến trình python nào khác dùng GPU | tranh GPU |
| S0.2 | Tag tính lại của A1s, B1, B2, B3 | bằng tag cũ, file `.pt` có | cấu hình bị lệch khi chép sang thư mục mới |
| S0.3 | **Canary:** nạp B3, đo bể chảy và `probe` bằng thư viện mới | rộng/sâu/dài 166/90/516 µm; E_res −7.4 ± 2 mJ; (tích + bay hơi)/nạp 0.805 ± 0.01 | thư viện đo/cổng sai |
| S0.4 | Bộ chỉ báo áp lên trường **giải tích R0** | khớp số R0 (522/122/86 µm) trong 2 µm | `indicator.py` sai |
| S0.5 | Hình học nêm y_a = 0: bản đồ vật liệu | s = 0.5 tại z = 60 µm ở y = 120 ± 10 µm; diện tích vùng đặc 27,000 µm² ± 3%; (200, 60) µm là đặc, (−100, 60) µm là bột | đặt sai vị trí nêm (lỗi +20 µm đã gặp) |
| S0.6 | MMS có nêm (`check_pde` với `wedge`) | ≤ 1e-10 | autodiff sai |
| S0.7 | `check_config`, `check_source` | không [WARN] | cấu hình không giải được |

**Tầng 1 — chi phí (GPU ≤ 15 phút, không cập nhật trọng số)**

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S1.1 | Đo 30 bước Adam: giây/bước, bộ nhớ GPU | ghi nhận; nếu > 0.25 s/bước thì kiểm lại trước khi chạy tiếp |
| S1.2 | Loss khởi điểm của B3 trên hình học nêm (chưa huấn luyện) | chỉ để biết (B1→B1k 14.95 trượt, B3 18.1 vẫn đạt, nên không phải ngưỡng cứng) |
| S1.3 | Profile một bước Adam (`torch.profiler`) | quyết định có làm tối ưu tốc độ (Forward Laplacian) hay không: chỉ làm nếu đạo hàm bậc hai > 50% thời gian bước **[ĐỀ XUẤT]** |

**Tầng 2 — khói đầu-cuối trên ngân sách tí hon (ước lượng ≤ 10 phút)**

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S2.1 | Adam 20 + L-BFGS float64 khối 5 vòng + resume + đánh giá, mọi file vào thư mục tạm | chạy hết; `ckpt` thật không đổi; tag khác tag thật |
| S2.2 | **Cổng vật lý trên checkpoint đã có** | `B3` **đạt**; `B1k_4d5bf24c` và `B1kc` **trượt**; `B1f1.4` trượt đúng gate E_res nhưng đạt T_min; bản cổng nhẹ cùng kết luận với `probe` đầy đủ |
| S2.3 | Giết tiến trình giữa khối rồi resume | mất ≤ 1 khối; giám sát phát hiện trong ≤ 3 phút |
| S2.4 | Khởi chạy hai tiến trình | tiến trình thứ hai từ chối |

**Tầng 3 — canary huấn luyện ngắn (ước lượng ≤ 15 phút)**

| ID | Kiểm | Tiêu chí |
|---|---|---|
| S3.1 | B3 → nêm y_a = 0, Adam 500 bước lr 1e-4, đo cổng nhẹ | loss giảm đơn điệu; không NaN; ước lượng thời gian cả lượt từ giây/bước đo được; T_min và E_res chỉ báo cáo |

Quy tắc: trượt bất kỳ mốc nào thì báo cáo và thảo luận, không chạy tầng tiếp theo.

## 5. Kế hoạch chạy chính (sau khi qua tầng 0–3)

| Bước | Việc | Ngân sách (ước lượng) | Phán quyết / quy tắc |
|---|---|---|---|
| **M0** | Chuẩn bị: tạo thư mục, `git init`, chép legacy, viết `runner/gates/indicator`, commit đăng ký trước | thời gian viết code | commit trước khi chạy |
| **M1** | Vệt thường: **không huấn luyện**; chỉ báo của B3 làm mốc so sánh | vài phút | S0.3 đã chứng minh pipeline đo |
| **M2** | `B3W0`: nêm y_a = 0, bột như đặc, vật lý B3, khởi động từ B3. Adam 3,000 bước lr 1e-4; L-BFGS float64 khối 500, tối đa 2,500 vòng; **dừng khi hai khối liên tiếp đạt cổng** (E_res ±10 mJ, T_min hụt ≤ 1%, tích/nạp ≤ 1) | ≤ 75 phút (Adam ~7–10 phút, L-BFGS ~25 phút nếu đủ 2,500 vòng) | đạt khi `check_physics` không [WARN] và cổng đạt; trượt thì dừng, báo cáo, không chạy lại |
| **M3** | Độ ổn định: đo bể chảy ở 50/70/90% thời gian quét; gấp đôi ngân sách L-BFGS; **tập điểm khác** (đổi seed Sobol) | ≤ 75 phút | chỉ báo ổn định giữa các lượt trong ngưỡng đăng ký **[ĐỀ XUẤT: 5 µm cho kích thước, 3 µm cho Δ độ sâu]** |
| **M4** | Chỉ báo overhang: Δ độ xuyên, tích tụ nhiệt, tốc độ nguội của `B3W0` so với B3 | vài phút | báo cáo kèm giới hạn |
| **M5** | Hiệu chỉnh hấp thụ (chỉ khi có dữ liệu, xem mục 6) | ≥ 3 lượt × ~25 phút | điều kiện đã chốt |
| **M6** | Kết khối: **chỉ mở lại khi có giả thuyết mới** | — | S2 đã trượt; không nằm trong kế hoạch này |

Tổng GPU cho M2–M4: khoảng 2.5–3 giờ (ước lượng); tầng 0–3: khoảng 1 giờ.

**Hai điều cần biết về M3:**
1. Khởi động ấm và Adam toàn lô là tất định, nên đổi seed khởi tạo **không** tạo biến thiên. Biến thiên chỉ đo được bằng cách đổi tập điểm huấn luyện, và `build_anchors` đang dùng seed Sobol cố định (thay đổi nhỏ trong thư mục mới).
2. Không có FD nên "sai số" của chỉ báo chỉ là độ nhạy với ngân sách và tập điểm, không phải sai số so với nghiệm đúng.

## 6. Hiệu chỉnh (M5) và một mâu thuẫn cần bạn quyết

Bạn đã chốt "một điều kiện cố định" và cũng chốt "hiệu chỉnh phải chia tập hiệu chỉnh/kiểm". Hai điều này xung đột: một điều kiện không chia được tập.
- (i) Chia theo **đại lượng**: hiệu chỉnh hấp thụ theo độ sâu, kiểm bằng độ rộng. Yếu, nhưng giữ một điều kiện.
- (ii) Dùng 2–3 điều kiện **chỉ để hiệu chỉnh/kiểm** (không tham số hóa). Mỗi điều kiện một lượt ~25 phút nếu hấp thụ cũng phải quét.
- (iii) Bỏ hiệu chỉnh, dùng hấp thụ đo (dữ liệu AM Bench) và báo độ lệch tuyệt đối như giới hạn.

Mốc test cho M5, không GPU: kiểm có tải được bảng số thực nghiệm (AM Bench 2022, Scime & Beuth) và điều kiện nào ở chế độ dẫn nhiệt (sâu/nửa rộng ≲ 1.2). Không có bảng số thì M5 dừng ở (iii).

## 7. Nếu chọn (A) thay cho (B)

Thêm trước M2: hoàn tất S1 (`B1f2`: chạy tiếp từ khối 500 bằng `--resume`; `B1f2.8`), rồi bước đổi P (kiểu B3P250, P chọn theo dữ liệu); thêm phép kiểm T_peak/độ nhất quán bay hơi. Mọi mốc test tầng 0–3 giữ nguyên, thay S0.3 bằng canary trên `B1f1.4` (ước lượng thêm 2–3 giờ GPU).

## 8. Rủi ro đã biết

- Nêm với bột như đặc cho Δ độ sâu khoảng +10 µm (số của P1 và FD cũ), nhỏ so với bất định của PINN. Chỉ báo có thể không tách khỏi nhiễu; M3 là phép kiểm đầu tiên.
- Bột như đặc bỏ qua tương phản dẫn nhiệt, tức đúng cơ chế vật lý của overhang. Kết quả chỉ là cận dưới, phải khai rõ; không thay thế được kết khối.
- P1 (`B3W`, y_a = +20) trượt `check_physics` ở T_min (hụt 1.28% so với ngưỡng 1%) vì L-BFGS float32 dừng sau 4 vòng; M2 chủ yếu kiểm xem L-BFGS float64 theo khối có sửa được không.
- Sập native không phụ thuộc thư viện; đã xử lý bằng resume và thử lại một lần.
