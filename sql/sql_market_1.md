


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

1321 餐馆营业额变化增长
customer_table:
customer_id, name, visited_on, amount

想分析一下可能的营业额变化增长（每天至少有一位顾客）

查询计算以 7 天（某日期 + 该日期前的 6 天）为一个时间段的顾客消费平均值


窗口函数
```sql
select visited_on, amount,average_amout
from(
    select visited_on,
    sum(total) over(order by visited_on rows between )
)


```

自连接
```sql
select a.visited_on, 
        sum(b.amount) amount, 
        round(sum(b.amout)/7,2) average_amount
from (select distinct visited_on from customer) a
join customer b on datediff(a.visited_on, b.visited_on) between 0 and 6
where a.visited_on >= (select min(visited_on) from customer) +6
group by a.visited_on

```

1126 查询活跃业务
events_table:
business_id, event_type, occurences
代表 业务id，事件，次数


如果一个业务的某个事件类型的发生次数大于此事件类型在所有业务中的平均发生次数，并且该业务至少有两个这样的事件类型，那么该业务就可被看做是活跃业务。

```sql

```


```sql

```


```sql

```
