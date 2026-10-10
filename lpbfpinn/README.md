# lpbfpinn

PINN nhiệt cho overhang không support, Ti-6Al-4V, LPBF. Theo `prereg/00_KE_HOACH_TI64.md`.

## Trạng thái

| Thành phần | Trạng thái |
|---|---|
| `materials.py` (bảng 2 Cook & Ritchie, Moser) | xong, có kiểm thử |
| `physics.py` (nhiệt dung, dẫn nhiệt, nguồn, bay hơi, biên trên) | xong, có kiểm thử |
| `stefan.py` (nghiệm Neumann) | xong, kiểm chứng cân bằng năng lượng và PDE |
| `gate_stefan.py` (cổng Stefan 1D bằng DeepXDE) | chạy được; kết quả trong `runs/gate_stefan/` |
| `manifest.py` (JSON mỗi lượt chạy) | xong |
| Bài overhang (hình học nêm, lịch sử pha vòng lặp ngoài, huấn luyện) | **chưa viết** |

Bài overhang chỉ chạy sau khi cổng Stefan được đánh giá và tiêu chí đã khóa trong `prereg/08`.

## Chạy

```bash
pip install torch deepxde pytest scipy
python -m pytest -q lpbfpinn/tests
DDE_BACKEND=pytorch python -m lpbfpinn.gate_stefan --iters 20000 --out runs/gate_stefan
```

## Giới hạn đã biết

- Lấy mẫu thích ứng gần T_sol–T_liq chưa được cài; cổng Stefan dùng lấy mẫu đồng đều.
- Tiêu chí đạt/không đạt của cổng chưa khóa; script chỉ ghi số liệu.
- Ngưỡng `T_cap = 6000 K` của bay hơi là tham số triển khai, chưa có trong prereg.
- Tính chất ngoài miền bảng 2 giữ giá trị biên.
