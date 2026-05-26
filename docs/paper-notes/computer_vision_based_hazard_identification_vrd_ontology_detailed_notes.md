# Computer Vision-Based Hazard Identification of Construction Site Using Visual Relationship Detection and Ontology

## Phạm vi tài liệu này

File này giải thích lại paper theo hướng chi tiết, tập trung vào:

- bài toán nghiên cứu
- framework tổng thể
- methodology
- dataset
- ontology
- thực nghiệm và kết quả
- hạn chế, hướng nghiên cứu tiếp theo, kết luận

Mình **lược bỏ related work và references** theo đúng yêu cầu.

## Thông tin bài báo

- Tên bài: `Computer Vision-Based Hazard Identification of Construction Site Using Visual Relationship Detection and Ontology`
- Tác giả: `Yange Li, Han Wei, Zheng Han, Nan Jiang, Weidong Wang, Jianling Huang`
- Venue: `Buildings`
- Năm: `2022`
- Số bài: `Buildings 2022, 12, 857`
- DOI: `https://doi.org/10.3390/buildings12060857`

## 1. Bài toán mà paper muốn giải quyết

Paper muốn cải thiện `systematic safety management` trên công trường bằng cách tự động phát hiện quan hệ giữa các object trong ảnh, rồi nối kết thông tin đó với tri thức an toàn.

Vấn đề paper nhấn mạnh là:

- object detection thông thường chỉ cho biết `có vật gì` và `ở đâu`
- nhưng hazard identification cần hiểu `các object đang tương tác như thế nào`
- ngoài ra còn tồn tại `semantic gap` giữa:
  - thông tin thị giác trích từ ảnh
  - thông tin ngôn ngữ trong safety rules / regulations

Ví dụ tiêu biểu của semantic gap trong paper:

- model vision nhìn thấy `helmet`
- con người và quy định an toàn lại nói ở mức khái niệm là `personal protective equipment`

Vì vậy, paper đề xuất một framework kết hợp:

- `visual relationship detection`
- `construction safety ontology`
- `natural language processing`
- `SWI Prolog` để suy luận logic

## 2. Ý tưởng trung tâm của paper

Paper dựa trên một ý tưởng khá rõ:

1. dùng `visual relationship detection` để biến nội dung ảnh thành các bộ ba quan hệ
2. dùng `ontology` để đưa các object cụ thể vào hệ khái niệm của an toàn xây dựng
3. dùng `NLP` để rút các luật an toàn từ văn bản thành dạng triple tương thích
4. dùng `SWI Prolog` để so sánh fact từ ảnh với rule từ quy định và suy ra risk

Triple mà paper dùng có dạng:

- `(object 1, relation predicate, object 2)`

Ví dụ:

- `(person, wear, helmet)`

Khi đưa vào Prolog, triple sẽ được đổi thành dạng:

- `wear(person, helmet)`

Đồng thời ontology sẽ cung cấp fact kiểu class mapping:

- `personal_protective_equipment(helmet)`

Từ đó máy có thể không chỉ thấy `helmet`, mà còn hiểu `helmet` là một loại `PPE`.

## 3. Điểm cần hiểu thật rõ trước khi đọc paper này

Paper mô tả một framework khá lớn, nhưng mức độ hoàn thiện của các khối **không đồng đều**.

### 3.1. Những gì paper thực sự làm khá rõ

- chọn và mô tả mô hình `VTransE`
- huấn luyện / đánh giá `VTransE` trên dataset `VRD`
- xây dựng thủ công một `construction safety ontology`
- mô tả cách ánh xạ visual triplets và ontology facts sang `SWI Prolog`

### 3.2. Những gì paper mới mô tả theo hướng framework / future work

- trích xuất triplet từ regulatory text bằng NLP
- tạo rules từ text để reasoning trong Prolog
- đánh giá end-to-end hazard identification trên dữ liệu construction thực tế

Nói cách khác:

- paper này **không phải** một hệ hazard checking end-to-end đã được kiểm chứng đầy đủ trên dataset construction riêng
- phần thực nghiệm của bài tập trung mạnh nhất vào `visual relationship detection`

Điểm này rất quan trọng nếu sau này chúng ta muốn dùng paper này làm nền so sánh hoặc tích hợp phương pháp.

## 4. Framework tổng thể

Framework được paper mô tả gồm các khối sau:

1. `Visual relationship detection`
   - phát hiện object
   - phát hiện relation giữa object
   - xuất ra triple `(subject, predicate, object)`
2. `Construction safety ontology`
   - biểu diễn tri thức miền an toàn xây dựng
   - ánh xạ object cụ thể sang class khái niệm
   - giải quyết semantic gap giữa vision và regulation text
3. `NLP-based entity relation extraction from safety regulations`
   - lấy rule text
   - phân tích phụ thuộc / dependency analysis
   - rút relation dưới dạng triple
4. `SWI Prolog reasoning`
   - đưa visual facts, ontology facts và rule facts vào Prolog
   - query / reasoning để suy ra risk và kiểm tra vi phạm

## 5. Methodology chi tiết

## 5.1. Khối 1: Visual Relationship Detection bằng VTransE

### 5.1.1. Vì sao paper chọn VTransE

Paper chọn `VTransE (Visual Translation Embedding network)` vì:

- đây là mô hình phát hiện relation end-to-end
- có thể phát hiện object và relation cùng lúc
- được xem là cạnh tranh trong nhóm visual relationship detection methods thời điểm đó

Paper cũng nhắc rằng VTransE đã cho kết quả tốt trên:

- `Visual Relationship`
- `Visual Genome`

### 5.1.2. Hai module chính của VTransE

Paper mô tả VTransE gồm 2 phần:

1. `object detection module`
2. `relation module`

Luồng chạy:

- ảnh đi vào object detection module
- mô hình phát hiện một nhóm object
- các object được đưa tiếp sang relation module
- đầu ra cuối là ảnh có object + các quan hệ dưới dạng triple `subject-predicate-object`

### 5.1.3. Cách biểu diễn relation bằng translation embedding

Paper dùng cảm hứng từ `TransE`.

Với quan hệ visual, họ xem triple ở không gian embedding như:

```text
subject + predicate ≈ object
```

Ví dụ trực giác paper nêu:

```text
person + wear ≈ helmet
```

Ký hiệu trong paper:

- `x_s`, `x_o`: đặc trưng M chiều của subject và object
- `t_p`: vector dịch chuyển cho predicate
- `W_S`, `W_O`: ma trận chiếu từ feature space sang relation space

Quan hệ được biểu diễn bằng:

```text
W_S x_s + t_p ≈ W_O x_o
```

Điểm quan trọng của cách làm này là:

- mô hình không chỉ đoán predicate như một nhãn phân loại rời rạc
- mà còn học một không gian nhúng nơi quan hệ là một phép dịch chuyển giữa subject và object

### 5.1.4. Hàm loss

Paper viết tổng loss:

```text
L = L_obj + 0.4 L_rel
```

Trong đó:

- `L_obj`: loss của object detection
- `L_rel`: loss của relation prediction

Relation loss được mô tả bằng softmax trên điểm số predicate:

```text
L_rel = Σ_(s,p,o)∈R -log softmax(t_p^T (W_O x_o - W_S x_s))
```

Ý nghĩa:

- mô hình tối ưu đồng thời phát hiện object và học predicate giữa object pairs

### 5.1.5. Cách tính điểm relation cuối cùng

Paper nói điểm cuối cho relation detection là tổng:

```text
S_s,p,o = S_s + S_p + S_o
```

Trong đó:

- `S_s`: điểm của subject
- `S_p`: điểm của predicate
- `S_o`: điểm của object

### 5.1.6. Feature extraction trong VTransE

Paper nói VTransE dùng 3 loại feature cho mỗi object trong relation:

1. `Classeme`
2. `Location`
3. `Visual feature`

#### a. Classeme

Là vector xác suất phân loại object.

Kích thước:

- `(N + 1)`
- gồm `N` class và `1` background

Vai trò:

- giúp loại các relation vô lý

Ví dụ paper nêu:

- `horse-drive-person` là quan hệ không hợp lý

#### b. Location feature

Là vector 4 chiều:

- `t_x`
- `t_y`
- `t_w`
- `t_h`

Ý nghĩa:

- `t_x`, `t_y`: translation bất biến theo scale
- `t_w`, `t_h`: biến đổi log-space cho width/height tương đối giữa subject và object

Paper viết công thức:

```text
t_x = (x - x') / w'
t_y = (y - y') / h'
t_w = log(w / w')
t_h = log(h / h')
```

với:

- `(x, y, w, h)` là box của subject
- `(x', y', w', h')` là box của object

#### c. Visual feature

Là feature lấy từ convolutional feature map cuối.

Paper nói:

- feature có dạng vector `D-d`
- được nội suy song tuyến tính từ feature map cuối

Điểm kỹ thuật quan trọng:

- paper dùng `bilinear interpolation`
- thay vì `ROI pooling`
- để giữ khả năng huấn luyện end-to-end với toạ độ khả vi

### 5.1.7. Ghép feature

Feature tổng của subject/object là weighted join của 3 nhóm trên:

- classeme
- location
- visual feature

Paper ghi:

```text
M = N + D + 5
```

và trọng số là các scaling layers có thể học được.

### 5.1.8. Kiến trúc object detector

Paper ghi:

- object detection network dùng `VGG-16`
- kế thừa từ `Faster R-CNN`

Chi tiết paper nêu:

- bỏ final pooling layer của VGG-16
- dùng feature map cuối có kích thước `W' x H' x C`
- `C = 512`
- `W' = [W / 16]`
- `H' = [H / 16]`

Relation error được back-propagate ngược vào object detector để cải thiện object feature cho relation learning.

### 5.1.9. Tối ưu

Paper chỉ nói:

- VTransE được train end-to-end bằng `stochastic gradient descent`

Paper **không nêu chi tiết sâu hơn** như:

- learning rate
- batch size
- số epoch
- scheduler

## 5.2. Khối 2: Construction Safety Ontology

### 5.2.1. Vai trò của ontology trong paper

Ontology được dùng để:

- biểu diễn tri thức an toàn xây dựng dưới dạng máy xử lý được
- nối ngữ nghĩa giữa object cụ thể trong ảnh và khái niệm trừu tượng trong safety rules
- chuẩn bị cho truy vấn và suy luận logic

Ví dụ rất quan trọng paper nêu:

- vision thấy `helmet`
- ontology gắn `helmet` là một instance / subclass thuộc `personal protective equipment`

Từ đó, rule ở mức `PPE` có thể gắn với object cụ thể trong ảnh.

### 5.2.2. Các thành phần ontology mà paper nhắc tới

Paper nhắc 5 thành phần thường gặp của ontology:

- classes
- relationships
- functions
- axioms
- instances

Tuy nhiên, paper nói trong thực tế không nhất thiết phải cứng nhắc tuân theo đầy đủ 5 primitive này.

### 5.2.3. Vì sao ontology được xây thủ công

Paper nói họ chọn xây ontology thủ công vì:

- không có ontology phù hợp sẵn để tái sử dụng
- ontology tái sử dụng thường cần sửa đổi nhiều
- chi phí chuyển đổi ontology cũ có thể lớn

### 5.2.4. Phương pháp xây ontology

Paper chọn nền tảng là `seven-steps methodology` của Stanford.

Nhưng họ có điều chỉnh cho phù hợp bài toán.

Các bước cuối cùng paper dùng là:

1. xác định domain và scope
2. liệt kê các term dùng phổ biến trong construction safety domain
3. phân loại các term và quan hệ phân cấp class hierarchy
4. định nghĩa properties của classes và facets của slots
5. tạo instances

Step `consider reusing existing ontologies` bị bỏ qua vì không tìm được ontology phù hợp.

### 5.2.5. Nguồn tri thức dùng để xây ontology

Paper nói họ dùng:

- `Safety Handbook for Construction Site Workers`
- kinh nghiệm trước đó của các học giả

### 5.2.6. Domain và phạm vi ontology

Ontology nằm trong:

- `construction domain`
- phục vụ `construction safety management`

Nó bao phủ các entity onsite xuất hiện trong regulations, gồm:

- `thing`
  - equipment
  - building structure
  - materials
- `personnel`
  - worker

### 5.2.7. Cách tổ chức class hierarchy

Paper dùng cách tiếp cận `top-down`.

Các top concepts trong sơ đồ ontology của paper xoay quanh `construction onsite object` và các nhóm con chính như:

- `loadshifting machinery`
- `suspended load`
- `personal protective equipment`
- `transport vehicle`
- `construction material`
- `construction workers`
- `excavating machinery`

### 5.2.8. Một số subclass cụ thể trong Figure 2

Từ mind map của paper, có thể đọc được các ví dụ subclass như sau:

#### a. Personal Protective Equipment

- ear protectors
- eye protectors
- gloves
- helmet

#### b. Loadshifting machinery

- bulldozer
- forklift trucks
- loader
- lorry
- truck

#### c. Suspended load

- attached wall lift
- hoist machine
- tower crane

#### d. Transport vehicle

- car
- motorcycle

#### e. Construction material

- brick
- cement
- glass
- lime
- linoleum
- sand
- steel
- stone
- tile
- wood

#### f. Excavating machinery

- backhoe loader
- muti-bucker excavator
- shovel
- single bucket excavator

#### g. Construction workers

- workers

### 5.2.9. Quan hệ trong ontology

Sơ đồ ontology của paper còn gợi ý một số relation vocabulary để mô tả tương tác giữa các nhóm object, ví dụ:

- `operate`
- `beneath`
- `contact`
- `wear`
- `drive`

Paper không formalize đầy đủ theo bảng schema như paper trước, nhưng có thể thấy ontology được thiết kế để hỗ trợ:

- mapping class hierarchy
- relation reasoning
- gắn detected object vào class phù hợp

### 5.2.10. Công cụ xây ontology

Paper dùng:

- `Protégé` để xây ontology
- `OWL` để biểu diễn ontology

Lý do:

- OWL có thể mô tả rõ terms, relationships và hierarchical structure

### 5.2.11. Tạo instance từ kết quả vision

Paper nói các entity detect được từ visual relationship detection sẽ được tạo thành `instances` trong ontology.

Đây là cầu nối quan trọng giữa:

- object cụ thể trong ảnh
- lớp khái niệm của ontology

## 5.3. Khối 3: NLP và SWI Prolog reasoning

### 5.3.1. Phần này paper mô tả ra sao

Paper nhiều lần nhắc tới:

- dùng NLP để trích entity relation từ safety regulation text
- dùng dependency analysis
- đưa rule vào SWI Prolog

Tuy nhiên cần nói thật rõ:

- paper **không trình bày chi tiết một module NLP hoàn chỉnh**
- paper **không đưa thí nghiệm định lượng riêng cho NLP**
- paper **không đưa kết quả end-to-end hazard inference trên dataset construction thực**

Phần này chủ yếu đóng vai trò:

- mô tả hướng framework
- mô tả cách logic reasoning sẽ được tổ chức
- nêu future work

### 5.3.2. Cách visual facts được đưa vào Prolog

Paper nói triple từ vision:

- `(object 1, relation predicate, object 2)`

sẽ được chuyển thành:

- `Relation(object1, object2)`

Ví dụ:

- `(person, wear, helmet)` -> `wear(person, helmet)`

### 5.3.3. Cách ontology facts được đưa vào Prolog

Paper dùng unary predicates để gán instance vào class ontology.

Ví dụ:

- `personal_protective_equipment(helmet)`

### 5.3.4. Vai trò dự kiến của NLP rule extraction

Paper nói trong nghiên cứu tiếp theo, họ sẽ:

- lấy safety regulations như `Safety Handbook for Construction Site Workers`
- dùng NLP / dependency analysis để trích relation triples
- đưa chúng sang cùng dạng relation trong Prolog

Tức là:

- visual facts và rule facts sẽ có representation tương thích
- ontology hỗ trợ semantic mapping ở giữa

### 5.3.5. Vai trò của SWI Prolog

`SWI Prolog` được dùng để:

- lưu facts
- lưu rules
- query logic
- thực hiện reasoning để suy ra whether onsite activities violate regulations

## 6. Dataset và dữ liệu mà paper dùng

## 6.1. Dataset cho visual relationship detection

Paper dùng benchmark:

- `VRD (Visual Relationships Dataset)`

Paper nêu các thống kê sau:

- `5000` images
- `100` object categories
- `70` predicates
- `37,993` relationships
- `6672` relationship types
- `24.25` predicates per object category

Split:

- `4000` images cho training
- `1000` images cho testing

### 6.1.1. Điểm rất quan trọng

Paper **không dùng một construction-specific visual relationship dataset tự xây dựng** cho phần thực nghiệm chính.

Đây là giới hạn đáng kể vì:

- VRD là benchmark quan hệ hình ảnh tổng quát
- không chuyên biệt cho construction safety

### 6.1.2. Điều paper thừa nhận

Paper tự thừa nhận rằng:

- việc xây dataset construction domain là tốn công và tốn thời gian
- chính vì vậy họ dùng VRD

## 6.2. Dữ liệu để xây ontology

Phần ontology không dùng dataset gán nhãn theo kiểu train/test.

Thay vào đó, nguồn tri thức là:

- `Safety Handbook for Construction Site Workers`
- kinh nghiệm học giả / domain knowledge

Tức là đây là phần:

- expert-driven
- manually constructed
- knowledge engineering

chứ không phải machine learning dataset theo nghĩa thông thường.

## 6.3. Dữ liệu cho NLP rule extraction

Paper có nhắc tới safety regulations text như nguồn đầu vào cho NLP.

Tuy nhiên paper **không công bố**:

- dataset text cụ thể
- số lượng văn bản / câu
- schema annotation
- train/validation/test split
- metric đánh giá NLP

Vì vậy phần này trong paper ở mức khái niệm nhiều hơn là module đã được đánh giá đầy đủ.

## 7. Thiết lập thực nghiệm

## 7.1. Cài đặt

Paper nói:

- thuật toán viết bằng `Python`
- cài đặt bằng `TensorFlow`

## 7.2. Phần cứng

Paper nói họ chạy trên máy có:

- `Nvidia Quadro series professional GPU card`
- `640 tensor kernels`

Paper không nêu model GPU cụ thể hơn.

## 7.3. Training / testing

Cho VTransE:

- train trên `4000` ảnh
- test trên `1000` ảnh

Paper không mô tả thêm:

- learning rate
- số epoch
- batch size
- data augmentation
- optimizer details ngoài việc trước đó VTransE dùng SGD

## 7.4. Metric đánh giá

Paper dùng:

- `Recall@100`

Paper định nghĩa:

- Recall@x là tỉ lệ quan hệ đúng xuất hiện trong top-`x` confident relationship predictions

Ngoài ra paper còn báo cáo một chỉ số `accuracy = 49.91%`.

Lưu ý quan trọng:

- cách mô tả accuracy trong paper khá ngắn và hơi khó đọc
- có vẻ đây là độ đúng khi predicate dự đoán top-1 trùng ground truth
- nhưng paper không formalize chỉ số này thật chặt như Recall@100

Vì vậy khi dùng lại paper, nên xem `Recall@100` là metric chính, còn `accuracy 49.91%` nên hiểu một cách thận trọng.

## 8. Kết quả thực nghiệm

## 8.1. Kết quả định lượng

Paper báo cáo:

- `accuracy = 49.91%`
- `predicate Recall@100 = 49.13%`

Theo paper, kết quả này cho thấy:

- VTransE cạnh tranh tốt trong nhóm visual relationship detection methods
- mô hình có thể dự đoán tương đối chính xác các tương tác giữa object

## 8.2. Kết quả định tính

Paper minh họa 5 ví dụ kết quả detection trên VRD.

### 8.2.1. Các ví dụ đúng

Paper đưa ra các relation như:

- `sky above person`
- `person wear helmet`
- `person on street`
- `person on the right of person`
- `cone next to street`
- `bush behind hydrant`
- `building behind bush`

Paper nói các quan hệ này bao gồm:

- `actions`
  - hold
  - wear
  - carry
- `geometry`
  - beneath
  - in front of
  - on
  - in

### 8.2.2. Vấn đề về biểu diễn relation tương đương

Paper chỉ ra một vấn đề rất hay:

- nhiều relation khác nhau có thể cùng mô tả một tình huống

Ví dụ:

- `building behind bush`
- `bush in front of building`

cả hai đều đúng với con người, nhưng với rule checking thì có thể gây rắc rối nếu safety rule chỉ định sẵn một cách diễn đạt.

Tương tự:

- `car next to car`
- `car behind car`

có thể đều xuất hiện như những mô tả hợp lý tùy góc nhìn.

Ý nghĩa:

- relation detection đúng về mặt thị giác chưa chắc đã tương thích ngay với rule representation
- cần ontology / semantic normalization / reasoning layer để giảm khác biệt diễn đạt

### 8.2.3. Các lỗi mà paper nêu

Paper cho một loạt ví dụ relation sai:

- `people wear phone`
- `sky in sky`
- `coat wear hat`
- `jacket wear hat`
- `dog wear glasses`

Paper giải thích nguyên nhân khả dĩ:

- các object ở quá gần nhau
- dễ sinh false relation matching
- knowledge transfer part cần cải thiện

### 8.2.4. Nhận định của paper về object detection và relation module

Paper kết luận:

- object detection module hoạt động hiệu quả
- relation prediction module khả thi nhưng chưa hoàn hảo

Đây là nhận xét khá hợp lý vì nhiều lỗi paper đưa ra là lỗi `object đúng nhưng relation sai`.

## 9. Ontology + Prolog mapping trong paper

Đây là phần có giá trị thực tiễn lớn nếu muốn tái sử dụng ý tưởng.

## 9.1. Dạng fact từ vision

Visual triple:

- `(person, wear, helmet)`

được chuyển thành fact Prolog:

- `wear(person, helmet)`

## 9.2. Dạng fact từ ontology

Ontology class mapping:

- `helmet` thuộc lớp `personal protective equipment`

được đưa thành:

- `personal_protective_equipment(helmet)`

## 9.3. Ý nghĩa của mapping này

Nó cho phép hệ thống nối:

- `object cụ thể` trong ảnh
- `class trừu tượng` trong safety rules

Ví dụ, nếu rule an toàn ở mức:

- workers must wear PPE

thì fact:

- `wear(person, helmet)`
- `personal_protective_equipment(helmet)`

có thể góp phần suy ra worker đang thoả một nhánh của yêu cầu PPE.

## 9.4. Nhưng paper chưa hoàn chỉnh bước reasoning end-to-end

Dù framework logic được mô tả, paper chưa cung cấp:

- bộ rule Prolog đầy đủ
- ví dụ hazard query hoàn chỉnh
- evaluation kết quả hazard reasoning trên construction scenes

Vì vậy đóng góp thực nghiệm thật sự của bài này vẫn chủ yếu nằm ở:

- VTransE relation detection
- ontology modeling
- ý tưởng tích hợp

## 10. Hạn chế do paper tự nêu

Paper nêu 2 nhóm hạn chế chính:

### 10.1. Chưa có construction-specific visual relationship dataset

Do chi phí cao, họ dùng `VRD` thay vì dataset construction domain.

Hệ quả:

- kết quả chưa phản ánh trực tiếp hiệu năng trên công trường thật
- dataset chuyên ngành có thể giúp tăng accuracy cho onsite tasks

### 10.2. Một số spatial relation khó mô hình hóa trong ontology

Paper nói các relation như:

- `in the front of`
- `on the right of`

khó mô tả và diễn giải gọn trong ontology model.

Do đó:

- các relation này khó dùng ở bước rule checking sau này

Paper gợi ý có thể dùng:

- `intersection over union`
- geometric features
- spatial features

để nhận diện các quan hệ kiểu:

- `within`
- `overlap`

### 10.3. Có thể mở rộng sang ergonomic analysis và posture

Paper đề xuất tương lai có thể thêm:

- ergonomic analysis
- posture detections

với các visual triplet như:

- `(worker, body_part, location)`

để mô tả motion của worker tốt hơn.

## 11. Future Research trong paper

Phần future work của paper thực ra rất quan trọng vì nó cho thấy chính xác khối nào chưa xong.

Paper nói nghiên cứu tiếp theo sẽ làm:

1. `triplet extraction from regulatory information`
   - dùng NLP
   - dùng dependency analysis
2. `transform safety rules into Prolog-compatible relation form`
   - `Relation(object1, object2)`
3. `semantic mapping through ontology`
   - class / subclass facts
4. `semantic reasoning in SWI Prolog`
   - kiểm tra hoạt động onsite có vi phạm regulation hay không

Nói ngắn gọn:

- paper hiện tại mới xây được nền `vision + ontology`
- paper định phát triển tiếp thành `vision + ontology + NLP rules + logic reasoning`

## 12. Kết luận ngắn gọn của paper

Paper kết luận rằng:

- visual relationship detection dựa trên deep learning có thể phát hiện nhiều tương tác giữa object trong ảnh
- `VTransE` cho kết quả khả thi với:
  - `accuracy = 49.91%`
  - `Recall@100 = 49.13%`
- ontology giúp thu hẹp semantic gap giữa object trong ảnh và khái niệm trong safety rules
- framework được đề xuất có tiềm năng cho automated hazard identification

Nhưng nếu đọc kỹ toàn bài, kết luận hợp lý nhất nên hiểu là:

- paper đã chứng minh tốt phần `relation detection` và `ontology concept`
- còn phần `hazard reasoning end-to-end` mới ở giai đoạn framework + hướng triển khai tiếp theo

## 13. Những chi tiết quan trọng nếu muốn tái sử dụng paper này sau này

Nếu sau này chúng ta muốn mô phỏng hoặc kết hợp phương pháp của paper này với các paper khác, có vài điểm rất đáng nhớ:

### 13.1. Paper này mạnh ở ý tưởng tích hợp hơn là validation end-to-end

Giá trị lớn nhất của paper nằm ở:

- dùng relation triple làm cầu nối
- dùng ontology để map semantic level
- dùng Prolog để reasoning

Chứ không phải ở việc đã chứng minh trọn vẹn một hazard detection system trên dữ liệu công trường.

### 13.2. VRD chỉ là benchmark tổng quát

Paper báo cáo metric trên:

- ảnh tổng quát
- object tổng quát
- predicate tổng quát

không phải:

- worker thật trên site
- PPE thật trên site
- hazard relations thật trên site

Vì vậy nếu dùng paper này cho construction safety, cần cẩn thận khi diễn giải kết quả.

### 13.3. Ontology là phần thực sự có giá trị chuyển giao

Từ Figure 2, ta thấy paper đã phác ra cấu trúc domain khá dùng được cho safety domain, gồm:

- machinery
- PPE
- materials
- workers
- vehicles
- suspended load

Đây là phần có thể tái sử dụng tốt nếu sau này chúng ta muốn:

- làm rule-based reasoning
- làm knowledge graph
- chuẩn hóa label giữa nhiều mô hình vision khác nhau

### 13.4. Relation normalization là bài toán lớn

Paper cho thấy một khó khăn thật:

- relation đúng ở mức thị giác nhưng diễn đạt khác nhau
- ví dụ `behind` và `in front of` ở hai chiều object

Nếu muốn làm rule checking nghiêm túc, ta có thể sẽ cần thêm:

- relation canonicalization
- ontology-level relation mapping
- symmetric / inverse relation handling

### 13.5. Paper không cung cấp đủ chi tiết để reproduce hoàn chỉnh framework cuối

Thiếu hoặc chưa rõ:

- pipeline NLP cụ thể
- dependency parsing setup
- bộ rule Prolog cụ thể
- ví dụ reasoning end-to-end hoàn chỉnh
- dataset construction domain để test hazard inference thực
- nhiều hyperparameter huấn luyện của VTransE

Vì vậy nếu tái lập, nhiều phần sẽ phải tự thiết kế lại.
