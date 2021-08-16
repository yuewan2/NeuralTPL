import re

if __name__ == "__main__":
    test_tokenlizer_map_data = '[C;D1;H3:4] - [c;H0;D3;+0:3] 1 : [cH;D2;+0:2] : [cH;D2;+0:1] : [c;H0;D3;+0:7] ( : [c:8] - [C:9] ( = [O;D1;H0:10] ) - [O;D1;H1:11] ) : [c:6] : [n;H0;D2;+0:5] : 1 >> <MASK> [C;H1;D2;+0:1] / [C;H1;D2;+0:2] = [C;H1;D2;+0:3] / [C;D1;H3:4] . [N;H2;D1;+0:5] - [c:6] : [c;H1;D2;+0:7] : [c:8] - [C:9] ( = [O;D1;H0:10] ) - [O;D1;H1:11]'
    test_tokenlizer_nomap_data = '[C;D1;H3] - [c;H0;D3;+0] 1 : [cH;D2;+0] : [cH;D2;+0] : [c;H0;D3;+0] ( : [c] - [C] ( = [O;D1;H0] ) - [O;D1;H1] ) : [c] : [n;H0;D2;+0] : 1 >> <MASK> [C;H1;D2;+0] / [C;H1;D2;+0] = [C;H1;D2;+0] / [C;D1;H3] . [N;H2;D1;+0] - [c] : [c;H1;D2;+0] : [c] - [C] ( = [O;D1;H0] ) - [O;D1;H1]'
    map_list = re.findall('\:([0-9]+)\]', test_tokenlizer_map_data)
    assert len(re.findall('\[[^\]]+]', test_tokenlizer_nomap_data)) == len(map_list)
    test_tokenlizer_nomap_data_split = test_tokenlizer_nomap_data.split(' ')
    test_tokenlizer_nomap_data_add_map = []
    map_cnt = 0
    for token in test_tokenlizer_nomap_data_split:
        if re.findall('\[[^\]]+]', token):
            token = token.replace(']', ':{}]'.format(map_list[map_cnt]))
            map_cnt += 1
        test_tokenlizer_nomap_data_add_map.append(token)

    print(' '.join(test_tokenlizer_nomap_data_add_map))
    assert ' '.join(test_tokenlizer_nomap_data_add_map) == test_tokenlizer_map_data
    print(map_list)
