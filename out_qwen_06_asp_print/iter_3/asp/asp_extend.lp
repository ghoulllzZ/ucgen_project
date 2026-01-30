ext("创建图层|extend|切换图层","创建图层","切换图层").
has_cond("创建图层|extend|切换图层").
ext("切换图层|extend|创建图形","切换图层","创建图形").

bad(K) :- ext(K,_,_), not has_cond(K), not has_ep(K).
#show bad/1.
