# Đăng ký trước 09 — mở rộng lấy mẫu điểm ra toàn bộ vùng sau chùm (giả thuyết H3 từ `prereg/08`)

Ngày: 2026-10-08 09:43 (đồng hồ). Người dùng đồng ý đăng ký hướng (1) trước: mở rộng lấy mẫu điểm collocation ra toàn bộ vùng sau chùm. **Commit file này trước khi sửa mã và trước khi chạy bất kỳ lượt nào.** Mọi thay đổi sau đó qua một mục sửa đổi có giờ lấy từ đồng hồ. Ngưỡng đánh dấu **[ĐỀ XUẤT]** do tôi đề xuất, người dùng quyết. Làm theo `CLAUDE.md`.

## 1. Câu hỏi và giả thuyết

`prereg/08` kết luận H1 (chưa hội tụ) đúng một phần: huấn luyện Adam thêm 20000 bước sửa được ΔD (lệch 8.46 → 0.77 µm) nhưng **không sửa vùng sau chùm** (trung bình T_PINN − T_FD ở vùng cách chùm hơn 16 r_b về phía sau, 70% quét: C3 −130 → **−152 K**, C4 −262 → **−232 K**), thiếu hụt năng lượng của C3 (−11%), thời gian lưu tuyệt đối của C3 (−104 µs) và ΔT_pk. Vùng này chiếm 28–54% sai số bình phương. Loss giảm theo bước^(−0.25), quá chậm để chạy thêm Adam làm lối ra (cần ~5700 lần số bước để về mức B3, suy diễn).

**Cơ chế lấy mẫu hiện tại** (`legacy/lpbf_pinn.py` mục 5.1, `config.SAMPLING`): `n_int_src = 2^18` điểm trong một nửa ellipsoid dưới mặt, tâm ở chùm tại thời điểm của điểm, bán kính **trước 2 r_b, sau 16 r_b**, ngang 4 r_b, sâu 3 r_b, trải đều theo thể tích trong ellipsoid (nửa trước và nửa sau mỗi nửa số điểm). Vùng cách chùm hơn 16 r_b phía sau (miền có chiều dài ~35 r_b ở cuối quét) chỉ có `n_int_unif = 2^16` điểm đều toàn miền 4D. Tất cả điểm được dùng ở mọi bước (full-batch).

**Giả thuyết H3:** sai số vùng sau chùm do **thiếu điểm lấy mẫu ở đó**. **Câu hỏi:** nếu kéo dài bán kính sau lên 36 r_b (phủ toàn bộ vùng sau chùm ở mọi thời điểm) và tăng số điểm theo tỉ lệ để giữ mật độ gần chùm, thì sai số vùng sau chùm và các Δ của bảng chỉ báo có cải thiện không?

**Lưu ý đã biết (bằng chứng ngược chiều):** lỗi cũng lớn trong vùng lấy mẫu dày (trung bình −80 K C3, −86 K C4 sau 08), nên H3 một mình có thể không đủ. Kết quả "không cải thiện" là kết quả hợp lệ và loại được H3.

## 2. Thiết kế (khóa)

Hai lượt, tiếp tục từ checkpoint của 08, giữ nguyên PDE, mặt nạ, loss và siêu tham số; **chỉ đổi lấy mẫu**:

| Ca | Bậc | Khởi tạo từ | Bao T_max cố định |
|---|---|---|---|
| C3 | `B3s` | `runs/F08/B3s_c/B3s_92803317.pt` | `runs/ENV/B3_iter0.pt` |
| C4 | `B3W0s` | `runs/F08/B3W0s_c/B3W0s_98431da4.pt` | `runs/Fs/B3W0s_it0/envelope_next.pt` |

- **Thay đổi lấy mẫu (nhánh W, "wake"):** `src_back` 16 → **36** r_b; `n_int_src` 2^18 → **2^19** (524288). Lý do số điểm: nửa sau của ellipsoid chiếm một nửa số điểm và thể tích tăng 36/16 = 2.25 lần; nhân đôi số điểm giữ mật độ nửa sau gần chùm ở ~0.89 lần cũ và nhân đôi mật độ nửa trước (không quan trọng). Mọi tham số lấy mẫu khác giữ nguyên (`src_front = 2`, `src_side = 4`, `src_depth = 3`, `z_power = 2`, `n_int_unif = 2^16`, `n_bnd_*`), cùng seed Sobol.
- **Huấn luyện:** Adam float32, `lr = 1e-4`, `λ_E = 1`, `λ_P = 1000`, seed 42, `--monitor-every 1000`, `--stop-gates 0`, `--max-iters 0`, `--adam-iters` **[ĐỀ XUẤT] 10000**, `--time-cap-s` **[ĐỀ XUẤT] 5400**. Số bước ít hơn 08 vì mỗi bước chậm hơn (ước tính 0.28–0.32 s/bước, chưa đo).
- **Nhánh đối chứng (A, có điều kiện):** cùng 10000 bước từ cùng checkpoint nhưng lấy mẫu mặc định (`src_back = 16`, `n_int_src = 2^18`, ~0.17 s/bước, ước ~30 phút mỗi ca). **Chỉ chạy nếu nhánh W đạt nhãn CẢI THIỆN hoặc ĐẠT** (mục 3), để tách ảnh hưởng của lấy mẫu khỏi ảnh hưởng của huấn luyện thêm. Nếu nhánh W không cải thiện thì không cần đối chứng (huấn luyện thêm không thể làm nhánh W tốt hơn mà đối chứng không tốt hơn cũng không giải thích được gì thêm).
- **Quy tắc dừng không dùng FD:** dừng khi hết 10000 bước, hết 5400 s, hoặc loss không hữu hạn. Một checkpoint cuối mỗi lượt, không chọn theo FD. Cùng quy trình an toàn như 08: kiểm tốc độ (đo s/bước trên 200 bước) và `nvidia-smi` trước khi chạy dài; GPU một tiến trình một lúc.
- **Thay đổi mã (tối thiểu, sau khi commit file này):** thêm tham số `--sampling` (chuỗi JSON ghi đè `SAMPLING`) vào `runner.py` và `config.build`; tag của lượt có băm phần ghi đè (để checkpoint không trùng tag). Không đổi hành vi khi không dùng `--sampling`. Kiểm: số điểm và phạm vi điểm sinh ra theo `src_back = 36` (smoke).

## 3. Đánh giá (khóa)

Một lần cho mỗi nhánh, sau khi cả hai ca xong: `experiments/pinn_cycles.py --tag f09`, `fd_compare.py compare --tag f09` (FD cố định: `runs/FD/fd_cases.json`, lưới 5 µm, hai cận p = 2 và p = 1, ngân sách như `prereg/07`), `fd_errormap.py` và `fd_fieldcheck.py` với `PINN_TAG=f09`.

**Chỉ tiêu chính (vùng sau chùm, 70% quét, `errormap`):** trung bình (T_PINN − T_FD) trên các ô T_FD > 600 K cách chùm hơn 16 r_b phía sau. Mốc 08: C3 −152.3 K, C4 −232.4 K. **Thành công** nếu \|trung bình\| giảm ít nhất một nửa ở **cả hai** ca (≤ 76 K và ≤ 116 K) **[QUY ƯỚC: 50%]**.

**Điều kiện không thoái lui (khóa):** (i) ΔD vẫn ĐÚNG ở `compare` (hai cận); (ii) lệch trung bình vùng quanh chùm không xấu hơn 1.25 lần mốc 08 (C3 ≤ 100 K, C4 ≤ 107 K, theo giá trị tuyệt đối) **[QUY ƯỚC: 1.25]**; (iii) hụt nhiệt độ âm của cổng đầy đủ (b) của C4 không xấu hơn 1.26%.

**Nhãn:**
- **ĐẠT:** ΔD, ΔT_pk, Δ thời gian lưu cùng ĐÚNG và thỏa điều kiện không thoái lui. (Tỉ số nguội vẫn KHÔNG XÁC ĐỊNH ở FD, không chặn.)
- **CẢI THIỆN:** chỉ tiêu chính thành công ở cả hai ca và thỏa điều kiện không thoái lui, nhưng chưa ĐẠT.
- **KHÔNG ĐỔI / XẤU HƠN:** còn lại.

**Chẩn đoán (chỉ báo cáo):** độ sâu / rộng / dài tự ghi, năng lượng lưu trữ so FD, phần dư PDE chuẩn hóa (trung vị, phân vị 90), trung bình T theo vùng và dọc đường, cổng đầy đủ, thời gian mỗi bước.

**Số lần thử:** mục này cho phép **một** lần đánh giá cho mỗi nhánh. Nếu CẢI THIỆN nhưng chưa ĐẠT hoặc nếu không đổi, bước tiếp theo (L-BFGS với mẫu cố định, sức chứa mạng) cần đăng ký 10 trước.

## 4. Dự đoán (ghi trước, có thể sai)

- Chỉ tiêu chính thành công ở cả hai ca (nhãn CẢI THIỆN trở lên): **~30%**. Lý do không cao: lỗi ở vùng quanh chùm cũng lớn; ở 08 thêm huấn luyện làm vùng sau chùm của C3 xấu đi.
- Nhãn ĐẠT: **~8%**; Δ lưu chuyển sang ĐÚNG: ~20%; ΔT_pk chuyển sang ĐÚNG: ~15%.
- ΔD vẫn ĐÚNG sau mở rộng lấy mẫu: **~65%** (rủi ro: gần chùm có thể kém đi khi trải điểm ra xa).
- Thiếu hụt năng lượng lưu trữ của C3 giảm xuống dưới 6%: ~35%.
- Tốc độ mỗi bước nằm trong 0.26–0.34 s: ~70%; một lượt không dùng hết 10000 bước trong 5400 s: ~20%.
- Phần dư PDE trung vị giảm: C3 ~50% (xu hướng nhẹ), C4 ~40%.

## 5. Giới hạn (khóa)

Một seed, một điều kiện, mặt nạ cố định (đúng FD), FD hai lưới 10 và 5 µm với giai đoạn B bỏ (A5). Chỉ trả lời việc lấy mẫu vùng sau chùm có sửa PINN cách nhiệt hay không; không nói về vật lý. Nhánh W thay đổi đồng thời bán kính sau và số điểm nên không tách riêng hai yếu tố (nếu có tác dụng, tách bằng 10). Kết quả CẢI THIỆN hoặc ĐẠT trên một seed chưa nói về độ lặp lại.


---

## Kết quả 09 (2026-10-08 11:54, đồng hồ): mở rộng lấy mẫu vùng sau chùm KHÔNG sửa được sai số vùng sau chùm; nhãn KHÔNG ĐỔI / XẤU HƠN

**Huấn luyện (nhánh W, theo mục 2):** cả hai lượt chạy hết 10000 bước (C3 `B3s_16ad0fbb`: 2973 s, 0.297 s/bước; C4 `B3W0s_4bf339b8`: 3342 s, 0.333 s/bước), dừng vì ngân sách bước. Lấy mẫu mới: 626688 điểm (364544 ban đầu), 46357 điểm nhóm nguồn nằm cách chùm hơn 16 r_b (trước: 0), xa nhất 34.8 r_b. Cổng đầy đủ: C3 **đạt** (hụt nhiệt độ âm 0.96%, F08 0.72%); C4 **trượt cổng (b)** (hụt 1.61% > 1%, F08 1.26%), `check_physics` 1 cảnh báo. Loss (hạng phần dư PDE) C3 0.00498, C4 0.00840; **không so được trực tiếp với F08** vì tập điểm khác.

**Chỉ tiêu chính (vùng sau chùm, 70% quét; `errormap_f09.json`, FD 10 µm):**

| Ca | Mốc F08 | F09 (nhánh W) | Giảm | Mục tiêu (≥ 50%) |
|---|---|---|---|---|
| C3 | −152.3 K | **−152.4 K** | 0% | ≤ 76 K: **không** |
| C4 | −232.4 K | **−193.1 K** | 17% | ≤ 116 K: **không** |

**Chỉ tiêu chính không đạt ở cả hai ca.** Điều kiện không thoái lui: (i) ΔD vẫn ĐÚNG ✓; (ii) lệch vùng quanh chùm không quá 1.25 lần mốc: C3 −90.6 K (≤ 100 K ✓), C4 −94.4 K (≤ 107 K ✓); (iii) hụt cổng (b) của C4 không xấu hơn 1.26%: **vi phạm** (1.61%). **Nhãn: KHÔNG ĐỔI / XẤU HƠN.** Nhánh đối chứng (có điều kiện) **không chạy**, theo mục 2.

**Đánh giá chính thức (`fd_compare.py compare --tag f09`, FD cố định, hai cận p = 2 và p = 1):**

| Δ (nêm − vệt thường) | F08 (lệch) | F09 | FD | \|lệch\| F09 | tol | Phán quyết F09 (F08) |
|---|---|---|---|---|---|---|
| ΔD cách nhiệt (µm) | +26.01 (0.77) | **+25.97** | +26.77 | 0.80 | 1.83 | **ĐÚNG** (ĐÚNG) |
| ΔT_pk (K) | +161.4 (18.5) | +159.3 | +179.9 | 20.6 | 15.98 | **SAI** (KHÔNG KẾT LUẬN) |
| Δ thời gian lưu (µs) | +307.0 (76.7) | **+282.5** | +230.3 | 52.2 | 22.2 | **SAI** (SAI) |
| tỉ số nguội | 0.549 | 0.480 | không xác định | – | 0.0457 | KHÔNG XÁC ĐỊNH |

**Phán quyết bảng chỉ báo (C4 − C3): SAI.** Δ lưu gần FD hơn F08 (lệch 76.7 → 52.2 µs) nhưng vẫn SAI.

**Chẩn đoán (chỉ báo cáo; F08 trong ngoặc):**
- Độ sâu / rộng / dài (µm, tự ghi): C3 90.2 / 186.1 / 524 (90.4 / 186.5 / 514); C4 116.1 / 189.1 / 541 (116.4 / 193.8 / 587; FD 117.6 / 204.8 / 773): độ rộng và chiều dài của C4 **xấu đi**.
- Trung bình (T_PINN − T_FD): vùng quanh chùm C3 −90.6 K (−79.8), C4 −94.4 K (−85.7); vùng sâu C4 +89.2 K (+13.6). Dọc đường (dọc / ngang / dọc x): C3 −55 / −78 / −123 K (−58 / −83 / −114), C4 −64 / −92 / −130 K (−58 / −84 / −123). T tại t = 1500 µs ở điểm chu trình 50%: C3 1153 K (FD 1350), C4 1501 K (FD 1640).
- Phần dư PDE chuẩn hóa (trung vị / phân vị 90): C3 15.3% / 42.7% (14.1% / 43.1%), C4 20.9% / 70.7% (18.4% / 68.5%): không cải thiện.
- Năng lượng lưu trữ so FD: C3 123.0 so 138.9 mJ (−11.4%; F08 −11.0%), C4 142.8 so 149.3 mJ (−4.3%; F08 −3.7%). E_id: C3 −7.7 mJ, C4 −4.3 mJ.

**Chấm dự đoán (mục 4):**

| Dự đoán | Kết quả |
|---|---|
| chỉ tiêu chính thành công ở cả hai ca (~30%) | **sai** (0% và 17%) |
| ĐẠT (~8%); Δ lưu ĐÚNG (~20%); ΔT_pk ĐÚNG (~15%) | **sai** cả ba |
| ΔD vẫn ĐÚNG (~65%) | **đúng** |
| thiếu hụt năng lượng của C3 xuống dưới 6% (~35%) | **sai**: −11.4% |
| tốc độ mỗi bước trong 0.26–0.34 s (~70%) | **đúng**: 0.297 và 0.333 s/bước (phép đo thử 200 bước cho 0.385, cao hơn thực tế) |
| một lượt không dùng hết 10000 bước (~20%) | **sai**: cả hai dùng hết |
| phần dư PDE trung vị giảm: C3 ~50%, C4 ~40% | **sai**: tăng nhẹ (14.1 → 15.3%; 18.4 → 20.9%) |

**Kết luận về H3:** mở rộng lấy mẫu ra toàn bộ vùng sau chùm **không sửa** sai số vùng sau chùm: C3 không đổi (−152.4 so −152.3 K), C4 giảm 17% (−232 → −193 K), trong khi gần chùm và độ rộng của C4 hơi xấu đi. H3 **không được ủng hộ** như nguyên nhân chính. Cùng với 08 (huấn luyện thêm không sửa vùng sau chùm), có thể hai lần thử cho thấy lỗi ổn định và không đáp ứng với số bước hay mật độ điểm, nên nên nghiêng về nguyên nhân khác (sức chứa mạng, điều kiện hóa tối ưu, dạng loss). Về mặt logic, nghiệm FD thỏa PDE đến sai số rời rạc nên là một nghiệm có loss gần 0 về nguyên tắc; loss của PINN ở 5×10⁻³–9×10⁻³ vì vậy cho thấy PINN bị giới hạn bởi tối ưu hoặc sức chứa chứ không phải bởi dạng loss (suy luận, chưa kiểm bằng cách đo loss của nghiệm FD). Các hướng này cần đăng ký 10 trước.

**Giới hạn:** một seed; một lần đánh giá cho mỗi nhánh; nhánh W đổi đồng thời bán kính sau và số điểm; nhánh đối chứng không chạy nên không tách được ảnh hưởng của huấn luyện thêm; C4 trượt cổng (b) nên không đủ điều kiện làm "PINN của hồ sơ".
