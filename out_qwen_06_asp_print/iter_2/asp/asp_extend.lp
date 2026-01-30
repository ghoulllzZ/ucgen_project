ext("切换图层|extend|启动图形编辑器","切换图层","启动图形编辑器").
has_ep("切换图层|extend|启动图形编辑器").
ext("切换图层|extend|切换图层","切换图层","切换图层").
has_ep("切换图层|extend|切换图层").
ext("创建新图层|extend|创建新图层","创建新图层","创建新图层").
has_ep("创建新图层|extend|创建新图层").

bad(K) :- ext(K,_,_), not has_cond(K), not has_ep(K).
#show bad/1.
