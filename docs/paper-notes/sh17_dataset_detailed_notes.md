# SH17 Dataset Notes

## Phạm vi tài liệu này

File này tập trung gần như hoàn toàn vào `dataset` của paper `SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry`.

Phần methodology huấn luyện model chỉ nhắc ngắn gọn khi nó giúp hiểu dataset tốt hơn.

## Thông tin paper

- Tên: `SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry`
- Tác giả: `Hafiz Mughees Ahmad, Afshin Rahimi`
- arXiv: `2407.04590`
- Journal version trong repo README: `Journal of Safety Science and Resilience (2024)`

## 1. Dataset này dùng để làm gì

SH17 được tạo ra để phục vụ:

- phát hiện con người và PPE trong môi trường sản xuất / công nghiệp
- huấn luyện object detection model cho bài toán PPE compliance
- benchmark nhiều YOLO variants trên cùng một bộ dữ liệu

Điểm paper nhấn mạnh là:

- đa số dataset PPE cũ nghiêng mạnh về `construction` và thường chỉ tập trung vào `helmet`, `vest`, hoặc vài PPE đơn lẻ
- SH17 cố gắng bao phủ tốt hơn bối cảnh `manufacturing / industrial worker safety`

## 2. Quy mô tổng thể của dataset

Theo paper, SH17 có:

- `8,099` ảnh đã gán nhãn
- `75,994` object instances
- `17` lớp

Theo phần setup thực tế từ Kaggle/repo, cấu trúc dữ liệu đầy đủ cũng có:

- `8,099` file ảnh trong `images/`
- `8,099` file nhãn YOLO trong `labels/`
- `8,099` file metadata JSON trong `meta-data/`
- `8,099` file nhãn Pascal VOC XML trong `voc_labels/`

Tức là bộ dữ liệu phát hành không chỉ có ảnh và nhãn YOLO, mà còn kèm:

- metadata nguồn ảnh
- VOC labels để dễ chuyển đổi pipeline

## 3. Nguồn dữ liệu

Paper nói họ thu ảnh từ:

- `Pexels`

Lý do chọn Pexels:

- quyền sử dụng rõ ràng hơn so với crawl từ web search engine
- giảm xung đột giấy phép thường gặp khi scrape từ nhiều nguồn
- giúp dataset dễ chia sẻ công khai hơn

Các truy vấn được dùng để kéo ảnh bao gồm kiểu:

- `manufacturing worker`
- `industrial worker`
- `human worker`
- `labor`

Paper đánh giá tags của Pexels là khá hữu ích trong việc lọc ảnh liên quan.

## 4. Quy trình thu thập và lọc dữ liệu

Pipeline paper mô tả:

1. crawl / thu ảnh từ Pexels bằng nhiều query
2. loại ảnh trùng lặp
3. thu được khoảng `11,000` ảnh ban đầu
4. khoảng `26%` ảnh là `empty`
   - nghĩa là không chứa object thuộc target classes
5. loại các ảnh empty trong quá trình labeling
6. còn lại `8,099` ảnh trong tập cuối

Điểm paper nhấn mạnh:

- ảnh đến từ môi trường công nghiệp đa dạng trên toàn cầu
- nhằm giảm bias vùng miền và bias giới tính / chủng tộc

## 5. Quy trình annotation

Paper mô tả quy trình annotation khá nghiêm túc:

- `4` người tham gia gán nhãn
- `3` người đầu tiên gán nhãn ban đầu
- team lead kiểm tra và sửa lỗi
- một graduate student kiểm tra lần cuối để sửa mislabel còn sót

Tool được dùng:

- `DarkLabel`
- `LabelImg`

Paper nói việc chọn tool chủ yếu là sở thích cá nhân, không ảnh hưởng tới chất lượng annotation.

## 6. 17 lớp của SH17

Paper tổ chức 17 lớp theo hướng:

- vừa chứa `body parts`
- vừa chứa `PPE items`
- thêm một số object liên quan trực tiếp tới công việc như `tools`

Ý đồ là:

- downstream application có thể bỏ qua lớp không cần
- nhưng dataset vẫn đủ rộng để xử lý nhiều tình huống an toàn công nghiệp

## 6.1. Danh sách lớp và số lượng instance

| ID | Class | Additional tags | Instances | Ghi chú |
| --- | --- | --- | ---: | --- |
| 1 | `Person` | `male`, `female`, `children` | `13,802` | Phân loại theo dấu hiệu nhìn thấy được |
| 2 | `Head` | `-` | `11,985` | Mọi góc nhìn của đầu |
| 3 | `Face` | `-` | `8,950` | Chỉ coi là thấy mặt nếu thấy mũi |
| 4 | `Glasses` | `on`, `off`, `safety`, `vision` | `1,945` | Kính an toàn |
| 5 | `Face-mask-medical` | `on`, `off` | `669` | Khẩu trang y tế |
| 6 | `Face-guard` | `on`, `off` | `134` | Tấm che mặt |
| 7 | `Ear` | `-` | `7,730` | Gắn nhãn tai để phục vụ PPE vùng tai |
| 8 | `Earmuffs` | `on`, `off` | `318` | Over-ear headphones cũng bị gán vào đây |
| 9 | `Hands` | `-` | `15,850` | Tập trung vào tay để phục vụ PPE vùng tay |
| 10 | `Gloves` | `on`, `off` | `2,790` | Găng tay |
| 11 | `Foot` | `-` | `796` | Thấy chân khi không mang giày; mỗi chân được annotate riêng |
| 12 | `Shoes` | `on`, `off`, `safety`, `other` | `4,560` | Safety shoes và thick joggers vào `safety`; còn lại vào `other` |
| 13 | `Safety-vest` | `on`, `off` | `530` | Áo phản quang / safety vest |
| 14 | `Tools` | `on`, `off` | `4,647` | `on` nghĩa là tool đang ở trong tay; bút chì và laptop không tính là tool |
| 15 | `Helmet` | `on`, `off`, `white`, `red`, `black`, `yellow`, `blue` | `927` | Có tag màu mũ |
| 16 | `Medical-suit` | `on`, `off` | `155` | Đồ bảo hộ y tế |
| 17 | `Safety-suit` | `on`, `off` | `530` | Đồ bảo hộ an toàn |

## 6.2. Ý nghĩa của các tag `on/off`

Paper ghi rõ:

- `on` nghĩa là item có mặt và đang được người đeo / sử dụng đúng ngữ cảnh
- `off` nghĩa là item có trong scene nhưng không đang được người đeo / không ở trạng thái sử dụng cần thiết

Ví dụ:

- với `tools`, `on` nghĩa là tool đang ở trong tay
- nếu tool xuất hiện trong cảnh nhưng không nằm trong tay, nó được tag `off`

Điều này rất quan trọng vì SH17 không chỉ là detection dataset thuần object presence, mà còn chứa tín hiệu để phục vụ PPE compliance.

## 7. Đặc điểm ảnh và annotation

Paper cho biết:

- ảnh được giữ ở `native resolution`
- kích thước lớn nhất: `8192 x 5462`
- kích thước nhỏ nhất: `1920 x 1002`
- có cả ảnh `landscape` và `portrait`
- trung bình `9.38 instances / image`

Paper còn nhấn mạnh đây là dataset có rất nhiều object nhỏ:

- `39,764` annotations chiếm `< 1%` diện tích ảnh
- `59,025` annotations chiếm `< 5%` diện tích ảnh

Điểm này cực kỳ quan trọng nếu sau này dùng SH17 để train:

- bài toán không hề chỉ là detect person lớn ở foreground
- các lớp như `ear`, `earmuffs`, `glasses`, `helmet`, `tools` có thể rất nhỏ

## 8. Mất cân bằng lớp

Paper nói dataset mất cân bằng khá rõ.

Một số lớp nhiều nhất:

- `Hands`: `15,850` instances, khoảng `20.9%`
- `Person`: `13,802`
- `Head`: `11,985`
- `Face`: `8,950`
- `Ear`: `7,730`

Một số lớp ít nhất:

- `Face-guard`: `134`, khoảng `0.2%`
- `Medical-suit`: `155`
- `Earmuffs`: `318`
- `Safety-vest`: `530`
- `Safety-suit`: `530`

Tác động thực tế:

- benchmark trên SH17 sẽ bị ảnh hưởng mạnh bởi tail classes
- các lớp ít dữ liệu và nhỏ thường cho mAP thấp hơn rõ rệt

## 9. Demographic tags cho lớp Person

Ngoài bounding box của `Person`, paper còn gắn thêm tags theo dấu hiệu nhìn thấy được để kiểm tra bias.

Các tags:

- `male`
- `female`
- `children`
- nhóm bề ngoài như:
  - `White`
  - `Black`
  - `Brown`
  - `Asian`

Paper cũng cảnh báo:

- các tag này suy ra từ đặc điểm quan sát được
- nên có thể không hoàn toàn chính xác

Thống kê trong paper:

| Group | Male | Female | Children | Total |
| --- | ---: | ---: | ---: | ---: |
| White | 2432 | 2032 | 37 | 4501 |
| Black | 1098 | 776 | 7 | 1881 |
| Brown | 577 | 218 | 12 | 807 |
| Asian | 963 | 1272 | 52 | 2287 |
| Total | 5070 | 4298 | 108 | 9476 |

Lưu ý:

- tổng `9476` ở bảng demographic nhỏ hơn tổng `13,802` person instances
- điều này cho thấy không phải mọi person annotation đều có / đủ điều kiện để gắn demographic tag

## 10. Metadata đi kèm

Paper có hẳn appendix cho metadata.

Các trường metadata theo paper:

- `Unique Identifier`
- `Width and Height`
- `URL`
- `Photographer Name`
- `Photographer URL`
- `Photographer ID`
- `Average Color`
- `Source`
- `Liked`
- `Description`

Trong bản dữ liệu thực tế, mỗi ảnh có một file JSON riêng trong `meta-data/`.

Ví dụ thực tế cho thấy metadata chứa:

- `id`
- `width`
- `height`
- `url`
- `photographer`
- `photographer_url`
- `photographer_id`
- `avg_color`
- `src`
  - có nhiều variant URL như `original`, `large2x`, `large`, `medium`, `small`, `portrait`, `landscape`, `tiny`
- `liked`
- `alt`

Nghĩa là metadata thực tế còn giàu hơn bảng mô tả trong paper ở chỗ:

- có nhiều URL ảnh theo kích cỡ
- có mô tả `alt` chi tiết hơn

## 11. Train / test split

Paper dùng:

- `80%` train
- `20%` test

Thống kê paper cho biết:

- `train instances = 60,636`
- `test instances = 15,358`

Tổng đúng bằng:

- `75,994`

### 11.1. Split theo từng lớp

| Class | Train | Test |
| --- | ---: | ---: |
| face-guard | 110 | 24 |
| medical-suit | 114 | 43 |
| safety-suit | 195 | 45 |
| ear-mufs | 269 | 49 |
| safety-vest | 433 | 97 |
| face-mask-medical | 519 | 151 |
| foot | 610 | 149 |
| helmet | 773 | 154 |
| glasses | 1547 | 398 |
| gloves | 2261 | 529 |
| shoes | 3604 | 956 |
| tools | 3724 | 923 |
| ear | 6118 | 1612 |
| face | 7095 | 1855 |
| head | 9558 | 2427 |
| person | 11068 | 2734 |
| hands | 12638 | 3212 |

## 12. Cấu trúc file thực tế của dataset phát hành

Sau khi setup từ repo + Kaggle, dataset đầy đủ có các thành phần:

- `images/`
  - ảnh gốc từ Pexels
- `labels/`
  - nhãn theo định dạng YOLO
- `voc_labels/`
  - nhãn theo định dạng Pascal VOC XML
- `meta-data/`
  - metadata JSON cho từng ảnh
- `train_files.txt`
  - danh sách file train
- `val_files.txt`
  - danh sách file validation/test

### 12.1. Định dạng nhãn YOLO

Một file nhãn mẫu trong `labels/` có dạng:

```text
9 0.219531 0.339714 0.221354 0.395052
```

Tức là:

- `class_id`
- `x_center`
- `y_center`
- `width`
- `height`

theo chuẩn YOLO normalized coordinates.

### 12.2. Định dạng Pascal VOC

`voc_labels/` chứa XML kiểu:

- `filename`
- `size`
- `object`
- `name`
- `bndbox`

Điều này giúp dataset dễ dùng cho nhiều pipeline hơn, không bắt buộc chỉ YOLO.

### 12.3. Điểm hơi lệch giữa paper/repo và file thật

Repo README nói nhiều chỗ là `test_files.txt`, nhưng dataset tải thực tế lại có:

- `train_files.txt`
- `val_files.txt`

Vì vậy khi dùng script hoặc YAML, nên bám vào file thật trong dataset hơn là ví dụ text trong README.

## 13. Setup thực tế đã làm trong máy này

Dataset đã được setup tại:

- `E:\data\SH17`

Các file tiện dùng thêm mình đã tạo:

- `E:\data\SH17\train_files_fullpath.txt`
- `E:\data\SH17\val_files_fullpath.txt`
- `E:\data\SH17\sh17.local.yaml`

Mục đích:

- `train_files.txt` và `val_files.txt` gốc chỉ chứa tên file ảnh
- YOLO thường cần list path đầy đủ
- `sh17.local.yaml` trỏ thẳng tới các file full path để dùng ngay với Ultralytics

## 14. Mapping class trong YAML của repo

Repo dùng mapping class như sau:

```yaml
0: person
1: ear
2: ear-mufs
3: face
4: face-guard
5: face-mask
6: foot
7: tool
8: glasses
9: gloves
10: helmet
11: hands
12: head
13: medical-suit
14: shoes
15: safety-suit
16: safety-vest
```

Lưu ý:

- tên trong YAML không hoàn toàn giống hệt tên bảng paper
- ví dụ:
  - `Face-mask-medical` trong paper thành `face-mask`
  - `Tools` thành `tool`
  - `Earmuffs` thành `ear-mufs`

Khi train/eval, nên bám vào thứ tự class của label files và YAML đang phát hành.

## 15. Benchmark có liên quan tới dataset

Vì người dùng hiện tại cần tập trung vào dataset hơn methodology, chỉ cần giữ lại các ý chính sau:

- paper benchmark nhiều model YOLOv8, YOLOv9, YOLOv10 trên SH17
- model tốt nhất trong paper là `YOLOv9-e`
- kết quả tổng thể tốt nhất:
  - `P = 81.0`
  - `R = 65.0`
  - `mAP50 = 70.9`
  - `mAP50-95 = 48.7`

Điều này cho thấy:

- dataset đủ lớn và đủ khó để tạo khoảng cách hiệu năng rõ ràng giữa các model
- các class nhỏ / ít mẫu vẫn còn là challenge thật sự

## 16. Những điểm rất đáng chú ý nếu muốn dùng SH17 sau này

### 16.1. Đây là dataset mạnh về PPE công nghiệp tổng quát hơn construction-only

SH17 hữu ích nếu mình muốn:

- nghiên cứu PPE detection ngoài bối cảnh công trường truyền thống
- bao phủ nhiều PPE hơn helmet/vest

### 16.2. Body-part classes là điểm đặc biệt

Các lớp như:

- `head`
- `face`
- `ear`
- `hands`
- `foot`

cho phép làm các pipeline linh hoạt hơn:

- detect PPE trực tiếp
- hoặc kiểm tra PPE theo body region

### 16.3. Dataset rất nhiều small objects

Nếu mình train detector trên SH17, cần kỳ vọng:

- small-object detection là nút thắt thật
- augmentation, input size, architecture choice sẽ ảnh hưởng mạnh

### 16.4. Class imbalance là vấn đề lớn

Những lớp rất ít như:

- `face-guard`
- `medical-suit`
- `earmuffs`

gần như chắc chắn cần:

- class-aware sampling
- focal-style losses
- data augmentation phù hợp
- hoặc train specialized heads nếu muốn tối ưu riêng

### 16.5. Metadata là tài sản rất đáng giá

Vì có metadata JSON cho từng ảnh, SH17 có thể được dùng không chỉ để train detector mà còn để:

- phân tích nguồn ảnh
- lọc subset theo mô tả / nguồn / kích thước
- xây pipeline multimodal hoặc data auditing

### 16.6. Repo GitHub không phải là full dataset release

Repo GitHub chủ yếu chứa:

- README
- script tải ảnh
- list URL
- helper scripts
- YAML

Full dataset thực tế nằm trên Kaggle.

Điểm này rất quan trọng cho reproducibility:

- clone repo xong chưa phải là đã có dataset
- muốn dùng được ngay vẫn cần tải bộ dữ liệu đầy đủ từ Kaggle

## 17. Tóm tắt ngắn gọn

Nếu phải tóm SH17 trong vài dòng:

- SH17 là dataset PPE/human safety dành cho môi trường công nghiệp, không chỉ construction
- nó có `8,099` ảnh, `75,994` instances, `17` lớp
- ảnh lấy từ `Pexels`, giữ `native resolution`, rất nhiều object nhỏ
- dữ liệu phát hành khá đầy đủ: ảnh, YOLO labels, VOC labels, metadata JSON, file split
- điểm mạnh là phạm vi PPE rộng hơn nhiều dataset cũ
- điểm khó là class imbalance và small-object density rất cao
