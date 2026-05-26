# Automatic Construction Hazard Identification Integrating On-Site Scene Graphs with Information Extraction in Outfield Test

## Phạm vi tài liệu này

File này giải thích lại paper theo hướng chi tiết, tập trung vào:

- bài toán và mục tiêu nghiên cứu
- framework tổng thể
- methodology
- dataset
- training setup
- experiments, results, case study
- limitations, future work, conclusions

Mình chủ động **lược bỏ phần related work và references** theo đúng yêu cầu.

## Thông tin bài báo

- Tên bài: `Automatic Construction Hazard Identification Integrating On-Site Scene Graphs with Information Extraction in Outfield Test`
- Tác giả: `Xuan Liu, Xiaochuan Jing, Quan Zhu, Wanru Du, Xiaoyin Wang`
- Venue: `Buildings`
- Năm: `2023`
- Số bài: `Buildings 2023, 13, 377`
- DOI: `https://doi.org/10.3390/buildings13020377`

## 1. Bài toán mà paper muốn giải quyết

Paper xử lý bài toán **tự động phát hiện nguy cơ an toàn trong bối cảnh outfield test**.

Trong bài, `outfield test` được mô tả là bối cảnh các thiết bị điện tử quân sự rời khỏi nơi nghiên cứu/sản xuất ban đầu để được đưa tới hiện trường phục vụ thử nghiệm thực tế. Trong quá trình đó phát sinh nhiều rủi ro an toàn, đặc biệt ở các giai đoạn:

- đóng gói và gia cố thiết bị trước khi vận chuyển
- lắp đặt và dựng thiết bị
- giai đoạn test sản phẩm
- tháo dỡ sau test

Ý tưởng cốt lõi của paper là:

- chỉ dùng ảnh/video giám sát thì chưa đủ vì thiếu hiểu biết ngữ nghĩa ở mức quy định an toàn
- chỉ dùng văn bản quy định thì không biết hiện trường đang diễn ra gì
- cần đưa **quy định an toàn** và **cảnh hiện trường** về cùng một dạng biểu diễn ngữ nghĩa, rồi so khớp chúng với nhau

Vì thế tác giả đề xuất một framework đa mô thức:

- trích xuất thông tin quan hệ từ **văn bản quy định an toàn**
- trích xuất thông tin quan hệ từ **ảnh hiện trường**
- so khớp hai phía để kiểm tra tuân thủ PPE và suy luận hazard

## 2. Ý tưởng trung tâm của paper

Paper thống nhất cả thông tin text và image về cùng một dạng:

- **relational triple** hay bộ ba quan hệ: `<Subject, Predicate, Object>`

Ví dụ:

- từ văn bản: `<worker, be equipped with, helmet>`
- từ ảnh: `<worker_0, wear, hard hat>`

Sau khi cả hai mô thức đều được đưa về dạng triple, hệ thống có thể:

- truy xuất các quy định liên quan đến một loại công việc
- đọc ảnh hiện trường để nhận ra người và PPE/hành động liên quan
- so khớp triple từ text và triple từ ảnh
- kết luận:
  - tuân thủ
  - không tuân thủ
  - không áp dụng

Đây là điểm quan trọng nhất của paper: **thu hẹp semantic gap giữa text và vision bằng biểu diễn triple chung**.

## 3. Mục tiêu nghiên cứu và thiết kế nghiên cứu

### 3.1. Mục tiêu nghiên cứu

Paper muốn xây dựng một framework có thể:

- tự động phân tích văn bản quy định an toàn
- tự động phân tích cảnh hiện trường
- tự động đối chiếu hai phía để kiểm tra an toàn

### 3.2. Thiết kế nghiên cứu

Paper chia thành 3 cụm công việc lớn:

1. `Data processing`
   - xây dựng dataset văn bản quy định an toàn tiếng Trung bằng gán nhãn thủ công theo schema
   - xây dựng dataset ảnh hiện trường bằng thu thập ảnh nguồn mở rồi gán nhãn thủ công
   - dùng thêm public benchmark để kiểm chứng mô hình
2. `Model development`
   - xây mô hình BERT-based để trích xuất thông tin từ text
   - xây mô hình scene graph generation để trích xuất thông tin quan hệ từ ảnh
   - thiết kế quy trình so khớp triple để safety checking
3. `Model validation`
   - dùng cả đánh giá định lượng và định tính trên dataset tự xây dựng và dataset công khai

## 4. Framework tổng thể

Framework của paper có 3 khối:

1. `BERT-based information extraction module`
   - đọc quy định an toàn
   - xuất ra các relational triple từ text
2. `On-site scene parsing module`
   - đọc ảnh hiện trường
   - phát hiện object và relation
   - sinh scene graph / visual relational triple
3. `Automatic safety checking process`
   - so khớp triple của text và triple của image
   - kiểm tra PPE compliance
   - suy luận hazard

Chức năng mà tác giả tuyên bố framework cung cấp:

- tự động parse văn bản quy định
- tự động sinh on-site scene graph
- tự động safety checking và hazard inference

## 5. Methodology chi tiết

## 5.1. Khối 1: BERT-based Safety Regulations Information Extraction

### 5.1.1. Bài toán của module này

Module này xử lý **joint entity-relation extraction** trên văn bản chuyên ngành an toàn lao động tiếng Trung.

Mục tiêu là trích xuất nhiều triple từ một câu quy định. Triple có dạng:

- `<S, R, O>`

Trong đó:

- `S`: subject entity
- `R`: predicate / semantic relation
- `O`: object entity

Ví dụ trong paper:

- `<worker, wear, helmet>`

### 5.1.2. Pre-processing và schema design

Đây là phần cực kỳ quan trọng vì paper không làm IE theo kiểu mở hoàn toàn, mà làm theo **schema định nghĩa trước**.

#### a. Schema là gì trong paper này

Schema là tập các mẫu triple quan hệ, mỗi schema gồm:

- subject entity type
- predicate
- object entity type

Paper coi việc định nghĩa schema này tương đương với việc formalize domain knowledge.

#### b. Nguồn để xây schema

Tác giả nói schema được thiết kế từ:

- yêu cầu tiêu chuẩn hóa an toàn lao động của outfield test
- kinh nghiệm từ các nghiên cứu trước
- truy hồi thông tin công khai

#### c. Số lượng relation type

Paper nói họ chọn **8 relation types được dùng thường xuyên nhất** bằng cách phân tích text miền bài toán.

Lưu ý quan trọng:

- paper chỉ nói có 8 relation types
- paper **không liệt kê đầy đủ cả 8 loại** trong phần text trích được
- paper chỉ đưa một vài ví dụ minh họa trong Table 1

#### d. Ví dụ schema trong Table 1

| Subject Type | Predicate | Object Type | SPO Example |
| --- | --- | --- | --- |
| `person` | `be equipped with` | `PPEs` | `<worker, be equipped with, eye protection>` |
| `person` | `perform...operations` | `working operations` | `<worker, perform...operations, welding operations>` |
| `working operations` | `occurrence` | `occupational injuries` | `<welding operations, occurrence, burns>` |

#### e. Cách chuẩn bị dữ liệu text để huấn luyện

Sau khi có schema:

- họ lấy văn bản quy định
- làm sạch lỗi chính tả / lỗi spelling
- tách thành từng câu riêng biệt
- loại bỏ câu không liên quan hoặc vô nghĩa
- chọn các câu liên quan tới hazard
- gán nhãn entity và relation theo schema
- chuyển mỗi câu về tập các triple

### 5.1.3. Encoding layer dùng BERT

Paper dùng `BERT-Base Chinese` làm encoder cho câu đầu vào.

Paper mô tả BERT như sau:

- mô hình ngôn ngữ dựa trên nhiều lớp Transformer
- có embedding space đầu vào `W`
- có `N` khối Transformer giống nhau
- mỗi token đầu vào có:
  - one-hot vector `x_i`
  - positional embedding `p_i`

Biểu diễn đầu vào:

```text
h = W * x_i + p_i
```

Qua từng lớp Transformer:

```text
h_j = Transformer(h_(j-1)), j ∈ [1, N]
```

Đầu ra là chuỗi contextualized embeddings.

Paper cũng viết ra loss cho bài toán tagging đa nhãn:

```text
L_m = log(1 + Σ exp(s_i)) + log(1 + Σ exp(-s_j))
```

Trong đó:

- `s_i`: class score của negative class
- `s_j`: class score của positive class
- `K`: tập negative classes
- `L`: tập positive classes

Ý chính ở đây là:

- BERT dùng để encode ngữ cảnh câu quy định
- đầu ra của BERT được chuyển sang bài toán sequence tagging
- sequence tagging ở đây là multi-label sequence tagging vì một câu có thể chứa nhiều triple chồng lấp

### 5.1.4. BIEO tagging decoder

Đây là phần methodological đáng chú ý nhất của module IE.

Paper không dùng BIO đơn giản, mà dùng **BIEO**:

- `B`: Begin
- `I`: Inside
- `E`: End
- `O`: Outside

#### a. Mục đích của BIEO trong paper

Giúp:

- phân biệt vị trí token trong entity
- hỗ trợ trích xuất nhiều triple trong cùng một câu
- đặc biệt hữu ích với các câu có **overlapping patterns**

#### b. Overlapping patterns là gì

Paper xem một câu là thuộc overlapping pattern nếu:

- hai triple chia sẻ cùng một cặp entity
- hoặc hai triple có ít nhất một entity chồng lấp, dù không cùng cặp entity

Đây là lý do họ nhấn mạnh joint extraction bằng tagging scheme thay vì tách rời entity extraction và relation classification.

#### c. Cách gán tag

Với `|N|` predicate trong schema, paper tạo:

- `2|N|` tag loại `B`
- `2|N|` tag loại `E`
- `1` tag `I`
- `1` tag `O`

Tổng cộng:

```text
4|N| + 2 tags
```

Lý do là paper phân biệt tag cho:

- subject entity
- object entity
- và relation/predicate tương ứng

#### d. Ví dụ gán nhãn trong paper

Câu ví dụ:

`Workers should be equipped with face protection during welding operations`

Paper nói câu này chứa 2 triple chồng lấp:

- `<worker, perform...operations, welding operations>`
- `<worker, be equipped with, face protection>`

Ví dụ:

- token `face` là từ đầu của object entity `face protection`
- token này liên quan tới subject `worker`
- đồng thời gắn với predicate `be equipped with`
- vì vậy nó nhận tag kiểu `sub-P1-B`

Token cuối của entity `protection` sẽ nhận tag `sub-P1-E`.

Các token không thuộc triple mục tiêu nhận tag `O`.

#### e. Cách tạo triple ở đầu ra

Sau khi decoder dự đoán chuỗi tag:

- hệ thống ghép cặp tag `B` và `E`
- đối chiếu subject-tag và object-tag theo predicate
- từ đó khôi phục ra các relational triple

Theo paper, cách này cho phép trích được nhiều triple hơn trong các câu quy định có mức chồng lấp cao.

## 5.2. Khối 2: On-Site Scene Parsing Module

### 5.2.1. Mục tiêu

Khối này xử lý ảnh hiện trường để trích xuất:

- object
- relation giữa object
- scene graph

Paper dùng SPO triple để biểu diễn cả visual relation. Quan hệ trong ảnh có thể là:

- action-based
- comparative
- spatial

Nhưng ở dataset tự xây dựng của paper, quan hệ thực tế chủ yếu xoay quanh PPE và thao tác làm việc.

### 5.2.2. Hai thành phần chính của scene parsing module

Paper tách module này thành 2 phần:

1. `Semantic modeling`
2. `Visual network`

### 5.2.3. Semantic modeling

Paper dùng thông tin ngữ nghĩa của label object/relation để hỗ trợ dự đoán predicate.

#### a. Word embedding khởi tạo bằng fastText

Paper dùng:

- `pre-trained fastText`
- `two million word vectors`
- học trên `Common Crawl`

Động cơ của họ là:

- fastText biểu diễn word như bag of n-grams
- vì vậy giữ được thông tin hình thái học trong từ
- theo paper, tốt hơn random init và word2vec cho bối cảnh này

#### b. Dự đoán relation có điều kiện theo object labels

Paper học theo trực giác:

- object labels là tín hiệu rất mạnh để suy ra relation label

Họ tính phân phối thực nghiệm:

```text
p̂(predicate | subject_class, object_class)
```

Nghĩa là:

- nếu biết class của subject và object
- có thể ước lượng predicate nào dễ xảy ra nhất giữa chúng

Paper giả định phân phối này ở test tương tự train.

### 5.2.4. Visual network

Đây là phần xử lý hình ảnh thật sự.

#### a. Object detector

Paper dùng `Mask R-CNN` để phát hiện object và trích xuất ROI features.

Các feature chính:

- `f_sub`: ROI feature của subject
- `f_obj`: ROI feature của object
- `f_rel`: ROI feature của vùng tương tác giữa subject và object

#### b. Predicate feature từ vùng tương tác

Paper nói relation feature nên đến từ vùng tương tác của subject và object, nên họ xây:

- một nhánh CNN riêng cho predicate feature

Ý này kế thừa từ nghiên cứu scene graph trước đó.

#### c. MLP để sinh embedding

Các feature được đưa qua MLP để tạo hidden features:

- `h_s`
- `h_r`
- `h_o`

Embedding của subject và object:

```text
y_s = g(h_s^k)
y_o = g(h_o^k)
```

Trong đó:

- `g(.)` là activation function
- `i, j, k` là chỉ số node/layer trong MLP

#### d. Fusion subject-object-relation

Paper ghép embedding của subject và object với hidden relation feature:

```text
h_r^j = concatenate(y_s, h_r^(j-1), y_o)
```

Sau đó sinh relation embedding cuối:

```text
y_r = g(h_r^k)
```

#### e. Gộp semantic logits và visual logits

Đây là điểm kiến trúc quan trọng:

- một nhánh sinh `logit(p_sem)` từ semantic modeling
- một nhánh sinh `logit(p_vis)` từ visual network

Sau đó:

```text
p(predicate) = softmax(logit(p_sem) + logit(p_vis))
```

Tức là dự đoán predicate cuối cùng được tạo từ **hợp nhất điểm số ngữ nghĩa và điểm số thị giác**.

### 5.2.5. Ý nghĩa của scene graph trong paper

Scene graph giúp paper không dừng ở object detection đơn thuần.

Thay vì chỉ biết:

- có worker
- có hard hat

hệ thống còn biết:

- worker đang `wear` hard hat
- worker đang `holding` welding tool

Điều này rất quan trọng vì safety checking cần quan hệ, không chỉ cần object xuất hiện.

## 5.3. Khối 3: Automatic Work Safety Checking

### 5.3.1. Mục tiêu

Khối này đối chiếu:

- triple rút ra từ **quy định an toàn**
- triple rút ra từ **ảnh hiện trường**

để phát hiện sai lệch giữa công việc đang diễn ra và yêu cầu an toàn.

### 5.3.2. Luồng xử lý

Từ flowchart của paper, quy trình là:

1. lấy quy định an toàn liên quan tới outfield test
2. lấy ảnh/giám sát từ key work area
3. dùng IE module để trích ra key entities và relational triples từ text
4. dùng scene graph generation để trích object và visual relations từ ảnh
5. so khớp hai tập triple
6. nếu có deviation:
   - suy luận hazard
   - phát lệnh điều chỉnh/correction command cho safety manager và workers
7. lặp cho tới khi công việc hiện tại hoàn thành
8. tạo và lưu safety report

### 5.3.3. Logic PPE compliance checking

Paper định nghĩa:

- `Y_ppe`: tập textual PPE-related triples
- `Y_vis`: tập visual relational triples

Logic:

- nếu không detect được worker trong ảnh:
  - cảnh này `N/A` cho PPE compliance
- nếu `Y_ppe` là tập con của `Y_vis` và `Y_ppe` không rỗng:
  - PPE compliance là `Yes`
- nếu không có triple textual nào xuất hiện trong triple của worker, hoặc chỉ xuất hiện một phần:
  - PPE compliance là `No`

Paper viết điều kiện theo kiểu:

- `Y_ppe ⊂ Y_vis`, `Y_ppe ≠ ∅`  -> đạt yêu cầu
- `Y_ppe ∩ Y_vis = ∅` hoặc `Y_ppe ∩ Y_vis = A, A ⊂ Y_ppe` -> không đạt yêu cầu

Ý nghĩa thực tế:

- nếu quy định yêu cầu worker phải có đầy đủ PPE liên quan mà ảnh chỉ thấy một phần hoặc không thấy, hệ thống đánh dấu không tuân thủ

### 5.3.4. Hazard inference

Sau PPE compliance checking, paper dùng `conditional judgment` để suy luận hazard.

Ví dụ:

- nếu scene là `work at height`
- nhưng triple `<worker, be equipped with, hard hat>` không xuất hiện
- thì có thể suy ra nguy cơ `head injury from falls`

## 6. Dataset và cách xây dựng dữ liệu

Đây là phần cần chú ý nhất nếu sau này muốn mô phỏng hoặc kết hợp phương pháp.

## 6.1. Dataset cho Information Extraction

### 6.1.1. Nguồn dữ liệu

Paper thu thập quy định an toàn outfield từ:

- open network resources
- documents do enterprise phát hành

### 6.1.2. Pipeline tạo dataset

Pipeline paper mô tả:

1. làm sạch text
   - sửa typo
   - sửa spelling mistakes
2. tách full text thành từng câu riêng
3. xóa câu không liên quan hoặc vô nghĩa
4. theo schema đã định nghĩa trước, chọn ra các câu liên quan đến hazard
5. dùng `label studio` để gán nhãn entity và relation
6. sinh relational triples theo schema

### 6.1.3. Quy mô dataset

Paper cho biết:

- chọn `336` câu key sentences liên quan tới on-site hazards
- thu được tổng cộng `1218` relational triples

### 6.1.4. Train/validation split

- `269` câu cho training
- `67` câu cho validation

Tổng đúng bằng `336` câu.

### 6.1.5. Lưu ý quan trọng về dataset text

Paper không nói rõ:

- danh sách đầy đủ 8 relation types
- số lượng từng relation type
- số lượng entity type chi tiết
- guideline gán nhãn chi tiết

Nói cách khác, paper cho biết logic xây dựng dataset, nhưng **không công bố đầy đủ specification để tái lập hoàn chỉnh chỉ từ bài báo**.

## 6.2. Public text benchmark: DuIE

Paper dùng thêm `DuIE` để kiểm chứng hiệu quả IE model.

Thông tin paper nêu:

- đây là large-scale Chinese relation extraction dataset của Baidu
- có `210,000` câu
- phủ `49` predicate types
- tỷ lệ câu có overlapping pattern cao hơn các dataset như `NYT` và `WebNLG`

Tác giả dùng DuIE để cho thấy method của họ không chỉ chạy trên dataset nội bộ nhỏ, mà còn hoạt động được trên benchmark lớn và khó.

## 6.3. Dataset cho On-Site Scene Graph Generation

### 6.3.1. Nguồn và cách chọn ảnh

Paper không dùng trực tiếp một tập video nội bộ lớn được mô tả chi tiết, mà làm như sau:

1. chọn các PPE safety checking rules từ quy định outfield, dựa trên IE model ở phần trước
2. crawl ảnh hiện trường từ open image resources theo các safety rules đó
3. manually select ảnh phù hợp

### 6.3.2. Quy mô dataset ảnh

- `829` ảnh được chọn làm final candidates

### 6.3.3. Công cụ gán nhãn

Paper nói dùng:

- `LabelImg`
- `PySimpleGUI`

### 6.3.4. Cấu trúc annotation

Gồm 2 lớp annotation:

1. `entity annotation`
   - bounding box
   - box label
2. `relation annotation`
   - subject
   - object
   - predicate

Kết quả annotation được tổ chức theo `Visual Genome format` để huấn luyện mô hình.

### 6.3.5. Thống kê relation annotation

Paper báo cáo tổng cộng `2316` visual relation annotations.

Chi tiết Table 4:

| Subject | Predicate | Object | Instances |
| --- | --- | --- | --- |
| `Worker` | `Wearing` | `Hard hat` | `1141` |
| `Worker` | `Wearing` | `Eye protection` | `220` |
| `Worker` | `Wearing` | `Hand protection` | `364` |
| `Worker` | `Wearing` | `Face protection` | `169` |
| `Worker` | `Holding` | `Welding tool` | `422` |

Nhận xét rất quan trọng:

- dataset tự xây dựng của paper khá hẹp
- subject gần như tập trung vào `Worker`
- predicate thực tế chỉ thấy rõ `Wearing` và `Holding`
- object tập trung vào PPE và `Welding tool`

Vì vậy dataset này phù hợp cho PPE compliance checking, nhưng chưa phải scene graph dataset bao quát rộng toàn bộ hazard hiện trường.

## 6.4. Public image benchmark: VRD

Paper đánh giá thêm trên `VRD (Visual Relationship Detection)` dataset.

Thông tin paper nêu:

- `5000` images
- `100` object categories
- `70` predicate categories

Paper dùng VRD chủ yếu để cho thấy mô hình có thể parse cả các relation đa dạng hơn:

- spatial
- comparative
- action-based

Tuy nhiên trong phần kết quả của bài, tác giả **không đưa bảng số liệu định lượng chi tiết trên VRD** trong đoạn trích chính; họ chủ yếu nêu rằng mô hình cũng được đánh giá trên VRD và cho ví dụ định tính.

## 7. Cài đặt huấn luyện và metric đánh giá

## 7.1. Information Extraction model

### 7.1.1. Backbone / pretrained model

- `BERT-Base Chinese`
- cấu hình paper nêu:
  - `12-layer Transformer`
  - `768 hidden size`
  - `12 attention heads`
  - `110M parameters`

### 7.1.2. Hyperparameters

- learning rate: `1e-5`
- early stopping:
  - dừng nếu `F1` trên validation không tăng trong `10` epochs liên tiếp

Paper không nêu rõ:

- batch size
- số epoch tối đa
- optimizer cụ thể cho IE module

## 7.2. Scene graph generation model

### 7.2.1. Split dữ liệu

- `80%` training
- `20%` testing

### 7.2.2. Optimization

- basic learning rate: `1e-3`
- optimizer: `momentum SGD`

### 7.2.3. Backbone

Trong kết quả, paper báo cáo với backbone:

- `ResNeXt-101-FPN`

Paper mô tả detector gốc là `Mask R-CNN`, và phần kết quả object detection / scene graph generation được trình bày với backbone trên.

## 7.3. Metrics cho Information Extraction

Paper dùng:

- `Precision`
- `Recall`
- `F1`

Một triple chỉ được tính là đúng khi:

- cả 2 entity đúng
- predicate cũng đúng

Đây là tiêu chuẩn khá chặt.

## 7.4. Metrics cho object detection

Paper dùng bộ metric chuẩn theo Mask R-CNN:

- `AP`
- `AP50`
- `AP75`
- `APS`
- `APM`
- `APL`

Trong đó:

- `AP` là average precision lấy trung bình qua các ngưỡng IoU
- `AP50`, `AP75` là AP tại IoU 0.50 và 0.75
- `APS`, `APM`, `APL` là AP cho object nhỏ/vừa/lớn

## 7.5. Metrics cho scene graph generation

Paper theo cách đánh giá của Zellers et al., với 3 task:

1. `SGGen`
   - phải đoán subject, object và toàn bộ label trong ảnh
2. `SGCls`
   - đã biết ground-truth bounding boxes của subject/object
   - mô hình đoán class label của subject/object và predicate
3. `PredCls`
   - đã biết ground-truth boxes và labels của subject/object
   - mô hình chỉ đoán predicate

Metric:

- `Recall@K`
- paper nhắc tới `R@20`, `R@50`, `R@100`
- bảng kết quả trong bài trình bày `R@20`

## 8. Kết quả thực nghiệm

## 8.1. Kết quả Information Extraction

### 8.1.1. Trên DuIE và Regulations Dataset

Table 2 của paper:

| Task | DuIE Prec. | DuIE Rec. | DuIE F1 | Regulations Prec. | Regulations Rec. | Regulations F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Information extraction | `77.3%` | `82.1%` | `79.6%` | `80.1%` | `78.6%` | `79.3%` |

Ý nghĩa:

- mô hình IE trên dữ liệu quy định tự xây dựng có F1 `79.3%`
- trên benchmark DuIE vẫn giữ F1 `79.6%`
- điều này cho thấy method không chỉ overfit vào tập nội bộ

### 8.1.2. Phân tích định tính của IE

Paper nhấn mạnh đa số câu trong bộ quy định thuộc dạng overlapping patterns nên extraction khó hơn.

Hai ví dụ ở Table 3:

#### Ví dụ 1

Input:

`Workers should be equipped with face protection and hand protection to prevent burns when performing welding operations.`

Các triple được trích:

- `<workers, be equipped with, face protection>`
- `<workers, be equipped with, hand protection>`
- `<welding operation, occurrence, burn>`
- `<workers, perform...operations, welding operations>`

#### Ví dụ 2

Input:

`Workers working at height should be equipped with hard hats to prevent head injury from falls.`

Các triple được trích:

- `<workers, be equipped with, hard hats>`
- `<working at height, occurrence, head injury from falls>`
- `<workers, perform...operations, working at height>`

Điểm hay ở đây là:

- một câu quy định không chỉ cho ra PPE rule
- mà còn cho ra quan hệ giữa loại công việc và loại chấn thương có thể xảy ra

## 8.2. Kết quả object detection

Table 5:

| Task | AP | AP50 | AP75 | APS | APM | APL |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection | `0.768` | `0.841` | `0.783` | `0.703` | `0.652` | `0.800` |

Điểm cần nhớ:

- `AP50 = 84.1%` là con số tác giả nhấn mạnh nhiều nhất
- detector hoạt động tốt nhất với object lớn hơn so với object vừa/nhỏ

## 8.3. Kết quả scene graph generation

Table 6:

| Task | Score |
| --- | --- |
| `SGGen (R@20)` | `0.507` |
| `SGCls (R@20)` | `0.855` |
| `PredCls (R@20)` | `0.978` |

Diễn giải:

- `SGGen = 50.7%` là task khó nhất vì phải dự đoán đầy đủ object + relation
- `SGCls = 85.5%` cao hơn nhiều khi đã biết bounding box
- `PredCls = 97.8%` rất cao khi subject/object đã biết đầy đủ

Điều này cho thấy:

- bài toán khó chủ yếu nằm ở phát hiện đầy đủ toàn bộ scene graph
- khi object đã rõ, việc phân loại predicate của họ khá mạnh

## 8.4. Kết quả định tính của scene graph

Paper cho ví dụ trên:

- dataset tự xây dựng
- VRD dataset

Các quan hệ mà mô hình có thể sinh ra:

- `<worker, wear, hard hat>`
- `<worker, holding, welding tool>`
- `<person, standing behind, building>`
- `<roof, next to, building>`

Tức là trên VRD, mô hình thể hiện khả năng xử lý nhiều loại relation hơn so với self-built PPE dataset.

## 9. Case study về hazard identification

Đây là phần chứng minh framework hoàn chỉnh có chạy được end-to-end.

Paper chọn 2 scene:

- `work at height`
- `welding operation`

## 9.1. Scene 1: work at height

### 9.1.1. Visual relational triple từ ảnh

Paper minh họa ảnh hiện trường và cho ra:

- `<worker_0, be equipped with (wear), hand protection>`

Điều đáng chú ý:

- không có triple cho `hard hat`

### 9.1.2. Textual relational triple từ quy định

Input sentence:

`Workers working at height should be equipped with hard hats to prevent head injury from falls.`

Output textual triples:

- `<workers, be equipped with, hard hats>`
- `<working at height, occurrence, head injury from falls>`
- `<workers, perform...operations, working at height>`

### 9.1.3. Safety checking

PPE compliance checking:

- `<worker_0, be equipped with, hard hat>` -> `No`

Hazard inference:

- `<working at height, occurrence, head injury from falls>` -> `Yes`

Nói ngắn gọn:

- hệ thống phát hiện người làm việc trên cao nhưng không thấy hard hat
- từ đó suy ra nguy cơ chấn thương đầu do ngã

## 9.2. Scene 2: welding operation

### 9.2.1. Visual relational triples từ ảnh

- `<worker_0, be equipped with (wear), hand protection>`
- `<worker_0, be equipped with (wear), face protection>`

### 9.2.2. Textual relational triples từ quy định

Input sentence:

`Workers should be equipped with face protection and hand protection to prevent burns when performing welding operations.`

Output textual triples:

- `<workers, be equipped with, face protection>`
- `<workers, be equipped with, hand protection>`
- `<welding operation, occurrence, burn>`
- `<workers, perform...operations, welding operations>`

### 9.2.3. Safety checking

PPE compliance checking:

- `<worker_0, be equipped with, face protection>` -> `Yes`
- `<worker_0, be equipped with, hand protection>` -> `Yes`

Hazard inference:

- `<welding operation, occurrence, burn>` -> `No`

Ý nghĩa:

- worker có đúng PPE yêu cầu
- nên hazard tương ứng không bị kích hoạt

## 9.3. Ba loại nhãn đầu ra ở bước compliance checking

Paper nói hệ thống có thể xuất 3 loại:

1. `Yes`
   - worker tuân thủ quy định
2. `No`
   - PPE check không đạt
3. `N/A`
   - scene không áp dụng cho PPE checking, ví dụ không detect được worker

## 10. Thảo luận của tác giả

Paper tự đánh giá framework có 2 điểm mạnh chính:

1. Ở phía text:
   - dùng BERT encoder + BIEO tagging
   - xử lý tốt các câu có overlapping patterns
   - trích được nhiều textual relational triples có ích cho hazard identification
2. Ở phía image:
   - scene graph generation kết hợp semantic modeling
   - hiểu được visual relation tốt hơn object detection thuần túy
   - có thể nối với text bằng triple matching để safety inspection

Paper cũng nhấn mạnh:

- F1 của IE khoảng `79.3%` và `79.6%`
- `AP50` của detector là `84.1%`
- `SGGen R@20` là `50.7%`

Từ đó tác giả kết luận phương pháp có tính khả thi cho on-site safety inspection.

## 11. Hạn chế được paper tự nêu

Paper nêu khá rõ 2 hạn chế chính:

### 11.1. Dataset còn nhỏ

- deep learning cần nhiều dữ liệu
- việc xây text dataset và image dataset chuyên ngành tốn công
- 2 dataset tự xây dựng của paper chưa đủ lớn để bao quát phần lớn hazard trong outfield construction

Tác giả nói cần:

- mở rộng dữ liệu
- làm giàu annotation

### 11.2. Quan hệ thị giác còn thiên về action-based

Paper chủ yếu phát hiện:

- action-based visual relations

Nhưng chưa xử lý kỹ:

- geometric relations
- spatial relations ở mức đo khoảng cách/proximity

Paper cũng nói chưa làm:

- pose estimation dựa trên human skeleton

Trong khi pose estimation có thể cung cấp thông tin chính xác hơn về con người để hazard identification.

## 12. Hướng nghiên cứu tiếp theo

Paper muốn phát triển tiếp thành:

- **multi-modal construction knowledge graph**

Cách họ hình dung:

- textual triples và visual triples sẽ liên tục được tích hợp vào knowledge graph
- knowledge graph được cập nhật động từ nhiều nguồn dữ liệu
- từ đó có thể làm dynamic hazard prediction

Ngoài ra tác giả còn muốn phát triển:

- hệ thống gợi ý huấn luyện an toàn cá nhân hóa

dựa trên:

- feature learning từ knowledge graph

## 13. Kết luận ngắn gọn của paper

Tóm lại, paper chứng minh một pipeline 3 bước:

1. trích xuất quy định an toàn thành triple bằng `BERT + BIEO`
2. trích xuất cảnh hiện trường thành triple bằng `scene graph generation`
3. so khớp hai phía để kiểm tra PPE và suy luận hazard

Các kết quả chính:

- IE trên regulations dataset: `Prec 80.1 / Rec 78.6 / F1 79.3`
- IE trên DuIE: `Prec 77.3 / Rec 82.1 / F1 79.6`
- Object detection: `AP 0.768 / AP50 0.841 / AP75 0.783`
- Scene graph generation:
  - `SGGen R@20 = 0.507`
  - `SGCls R@20 = 0.855`
  - `PredCls R@20 = 0.978`

Kết luận chính của tác giả là:

- phương pháp có thể tự động kết nối quy định an toàn và hiểu biết thị giác hiện trường
- biểu diễn triple chung là chìa khóa để làm compliance checking
- framework khả thi cho PPE safety inspection và hazard inference

## 14. Những chi tiết quan trọng nếu muốn tái sử dụng paper này sau này

Nếu sau này chúng ta muốn mô phỏng hoặc kết hợp phương pháp của paper này với các paper khác, có vài điểm cần nhớ thật rõ:

### 14.1. Đây không phải một hệ thống hazard detection tổng quát hoàn chỉnh

Nó mạnh nhất ở bài toán:

- PPE compliance checking
- rule-based hazard inference từ PPE thiếu / không đúng

Chưa phải hệ thống end-to-end bao quát toàn bộ hành vi, khoảng cách, pose, nguy cơ không gian phức tạp.

### 14.2. Triple matching là lõi logic suy luận

Điều làm paper này khác object detection thông thường là:

- nó không kết luận hazard trực tiếp từ ảnh
- nó kết luận hazard qua bước:
  - `text rules -> triples`
  - `image scene -> triples`
  - `match -> compliance / non-compliance -> hazard inference`

### 14.3. Dataset tự xây dựng của phần vision khá hẹp

Self-built scene graph dataset chủ yếu xoay quanh:

- worker
- wearing PPE
- holding welding tool

Nếu áp dụng cho domain PPE detection rộng hơn, nhiều khả năng phải:

- mở rộng object classes
- mở rộng predicate classes
- bổ sung spatial relations
- bổ sung nhiều hazardous scenarios hơn

### 14.4. Phần text là domain-specific IE chứ không phải open IE

Phương pháp text phụ thuộc mạnh vào:

- schema định nghĩa trước
- relation types định nghĩa trước
- tagging dựa trên predicate inventory đó

Do đó nếu đổi domain hoặc đổi loại quy định, gần như chắc chắn phải:

- thiết kế lại schema
- gán nhãn lại dataset
- fine-tune lại IE model

### 14.5. Paper không cung cấp đủ chi tiết để tái lập 100%

Những chỗ paper không mô tả đầy đủ:

- full list của 8 relation types
- guideline annotation chi tiết
- batch size / epochs cho IE
- chi tiết training đầy đủ của scene graph branch
- quantitative table chi tiết trên VRD

Vì vậy nếu muốn reproduce sát bản gốc, nhiều khả năng vẫn phải tự đưa ra assumption ở một số phần.
