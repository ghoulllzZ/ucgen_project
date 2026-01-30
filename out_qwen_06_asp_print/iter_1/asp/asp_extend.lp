ext("创建新图层|extend|切换图层","创建新图层","切换图层").
has_cond("创建新图层|extend|切换图层").
ext("选择图形|extend|选择图形","选择图形","选择图形").
has_ep("选择图形|extend|选择图形").

bad(K) :- ext(K,_,_), not has_cond(K), not has_ep(K).
#show bad/1.
