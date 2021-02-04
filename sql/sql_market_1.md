


市场分析系列

users_table:
user_id, join_date, favorite_brand

orders_table:
order_id, order_date, item_id, buyer_id, seller_id

item_table:
item_id, item_brand


1) 找到每一个用户按日期顺序卖出的第二件商品的品牌
2) 查询每个用户的注册日期和在 2019 年作为买家的订单总数。
3)
4)
5)
6)



1159 市场分析
找到每一个用户按日期顺序卖出的第二件商品的品牌
得到用户卖出的第二件商品的品牌后需要和用户最爱的品牌比较


把过滤条件写在连接条件，可以减少一次子查询
```sql
-- 718 ms best
select user_id seller_id, 
case when u.favorite_brand=a.item_brand then 'yes' else 'no' end 2nd_item_fav_brand
from users u
left join(
    select seller_id, item_brand, rank() over(partition by seller_id order by order_date) `rank` from orders o
    inner join items i
    on o.item_id= i.item_id
) a
on u.user_id =a.seller_id 
and  `rank`=2 order by 1;


-- 830 ms
select user_id seller_id, if(favorite_brand=item_brand, 'yes', 'no')  2nd_item_fav_brand
from users u
left join(
    select seller_id, item_brand, rank()over(partition by seller_id order by order_date) as rk
    from orders o
    join items i on o.item_id=i.item_id
    order by seller_id, order_date
) t1
on u.user_id = t1.seller_id and t1.rk=2;


-- 785 ms having sum
select user_id seller_id, if(favorite_brand = item_brand, 'yes', 'no') 2nd_item_fav_brand
from users left join (
    select o1.seller_id, item_brand
    from orders o1 
    join orders o2
    on o1.seller_id = o2.seller_id

    join items i
    on o1.item_id = i.item_id
    group by o1.order_id
    having sum(o1.order_date > o2.order_date) = 1
) tmp
on user_id = seller_id
```

1158 
查询每个用户的注册日期和在 2019 年作为买家的订单总数。

```sql
-- find order_num sum at this year
-- then regit_date

-- 734 ms 94%
select u.user_id buyer_id, join_date, ifnull(b.cnt,0) orders_in_2019
from users u left join(
    select buyer_id, count(order_id) cnt from orders
    where order_date between '2019-01-01' and '2019-12-31'
    group by buyer_id
) b
on u.user_id=b.buyer_id;
```




```sql



```

```sql

```


```sql

```

```sql

```


```sql

```
