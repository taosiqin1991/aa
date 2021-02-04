



sales_table:
sale_id, product_id, year, quantity, price
增加 seller_id, buyer_id, sale_date, 

product_table:
product_id, product_name
增加 unite_price

```sql
rank() over(partition by product_id order by year) as ranking

```


1) 获取sales表中所有产品对应的product_name 以及该产品的所有售卖year 和price
同一product_name, 不同年份算两条及以上。
2) 按product_id 来统计每个产品的销售总量
3) 每个销售产品的 第一年 的 产品id、年份、数量和价格。
4) 查询2019年春季才售出的产品。即仅在[2019-01-01,2019-03-31]之间。
5) 查询购买了 S8 手机却没有购买 iPhone 的买家
6) 查询总销售额最高的销售者。如果有并列的，就都展示出来。
7)
8)
9)
10)
11)
12) 


获取sales表中所有产品对应的product_name 以及该产品的所有售卖year 和price
```sql
-- syntax error if remove ()
-- 970 ms
-- 935 ms
select product_name, year, price from sales 
join product using (product_id);

-- 1100 ms
select product_name, year, price from sales a
left join product b on a.product_id=b.product_id;

-- 996 ms
select product_name, year, price from sales a
inner join product b on a.product_id=b.product_id;
```

每个销售产品的 第一年 的 产品id、年份、数量和价格
只涉及sales表

(product_id,year)的数组匹配查找最小年份的语句，保证结果都是出自第一年当年
语句最后不能加group by product_id,因为当年可能多次售出，价格与销量不同(或者相同)，必须保留这些结果

这两句等价
year first_year
year as first_year
```sql
-- 817 ms defeat 94%
select product_id, year first_year, quantity, price
from sales where (product_id, year)
in( select product_id, min(year) from sales group by product_id);


```

1082 销售分析
查询总销售额最高的销售者，如果有并列的，就都展示出来。

all 和每一个进行比较（大于最大的或者小于最小的）
any 则是大于任何一个都可以（大于最小的，小于最大的）
```sql
-- 809 ms defeat 90%
-- 790 ms
select seller_id from sales 
group by 1
having sum(price) >= all (select sum(price) from sales group by seller_id);



-- 874 ms defeat 42%
select seller_id from sales
group by 1
having sum(price)=(select sum(price) from sales
                   group by seller_id order by 1 desc limit 1);
```

1083 销售分析
查询购买了 S8 手机却没有购买 iPhone 的买家


列转行，然后加总筛选
case when
count + if
```sql

-- 768 ms defeat 99%
select s.buyer_id from sales s 
join product p on s.product_id=p.product_id
group by s.buyer_id
having count(if(p.product_name='s8', true, null))>=1 and count(if(p.product_name='iPhone', true, null))=0;


-- 798 ms
select buyer_id from(
    select buyer_id, 
    case when p.product_name='s8' then 1 else 0 end s8,
    case when p.product_name='iPhone' then 1 else 0 end iph 
    from sales s left join product p on s.product_id=p.product_id
) a
group by buyer_id 
having sum(s8)>0 and sum(iph)=0;



-- 1004 ms defeat 11%
select distinct buyer_id from sales
where buyer_id in(select buyer_id from sales 
                where product_id=(select product_id from product where product_name='S8'))

and buyer_id not in(select buyer_id from sales
                    where product_id=(select product_id from product where product_name='iPhone'));

```


1084  销售分析
查询2019年Q1才售出的产品

```sql
-- 788 ms defeat 94%
select p.product_id, product_name from product p
inner join sales s on p.product_id=s.product_id
group by 1
having min(sale_date)>='2019-01-01' and max(sale_date)<='2019-03-31';

```



product_id 来统计每个产品的销售总量

```sql
-- 996 ms defeat 13%
select product_id, sum(quantity) total_quantity 
from sales group by product_id;

-- 824 ms defeat 88%
select p.product_id,sum(quantity) total_quantity 
from product p join sales s
on p.product_id=s.product_id
group by product_id;

-- 862 ms 
select product_id, sum(quantity) total_quantity 
from sales group by product_id
order by product_id asc;
```

查询2019年春季才售出的产品。即仅在[2019-01-01,2019-03-31]之间。

```sql
-- find all possible product
select product_id, product_name from product p
join sales s where p.product_id = s.product_id and s.sale_date in date(2019-01-01) and date(2019-01-31);

```

```sql

```


1251 平均售价

prices_table:
product_id, start_date, end_date, price

unitssold_table:
product_id, purchase_date, units

expect:
product_id, average_price

内连接 按产品分组 然后算价格
此题用left join 和 inner join 都ok
```sql
-- sum(price_in_date * unit) / sum(unit) 
select p.product_id, round( sum(p.price *u.units)/sum(u.units), 2) average_price
from prices p 
inner join unitssold u
on p.product_id=u.product_id and u.purchase_date between p.start_date and p.end_date
group by p.product_id;

```

1164 指定日期的产品价格
product_table:
product_id, new_price, change_date

查找在 2019-08-16 时全部产品的价格，假设所有产品在修改前的价格都是 10。

1) 在此日期之前的产品的最后更改价格。
2) 所有产品，如果未更改，则原价格
```sql
-- 305 ms
select * from

(select product_id, new_price price from products
where (product_id, change_date) in
    (select product_id, max(change_date) from products 
    where change_date<='2019-08-16' group by product_id)

union
select distinct product_id, 10 price from products
where product_id not in 
    (select product_id from products 
    where change_date<='2019-08-16')

) tmp
order by price desc;

```


```sql
-- price_under_id_and_date
-- 290 ms  defeat 96%
-- 305 ms
select p.product_id, new_price price from products p
join (
    select distinct product_id,max(change_date) latest from products
    where change_date<='2019-08-16' group by product_id
) a
on p.product_id = a.product_id and p.change_date=a.latest

union
select distinct product_id,10 price from products
group by product_id
having min(change_date)>'2019-08-16'

order by product_id;

```

rank是上顿号，非引号。
```sql
-- 352 ms
select distinct p.product_id, ifnull(a.new_price, 10) price from products p
left join(
    select product_id, new_price, rank() over(partition by product_id order by change_date desc) `rank`
    from products where change_date<='2019-08-16'
) a
on p.product_id =a.product_id and `rank`=1;

```

```sql
-- 302 ms
-- 371 ms
select distinct p.product_id, ifnull(a.new_price, 10) price
from products as p
left join (select product_id, new_price from products
           where change_date <= '2019-08-16'
           and (product_id, change_date) in (select product_id, max(change_date)
                                             from products
                                             where change_date <= '2019-08-16'
                                             group by 1)) as a
on p.product_id = a.product_id;

```

1045 买下所有产品的客户

customer_table:
customer_id, product_id

product_table:
product_key

同一语句，执行时间也差异巨大。

```sql
-- 364 ms
-- 414 ms
select customer_id from customer
group by customer_id having count(distinct product_key) in
(select count(distinct product_key) from product p);

-- 375 ms
-- 401 ms
select customer_id from customer
group by 1 having count(distinct product_key)= (select count(product_key) from product);


```

586 订单最多的客户
orders_table:
order_num, customer_number, order_date, required_date, shipped_date, status, comment


```sql
-- 337 ms 97%
select customer_number from orders
group by customer_number order by count(*) desc limit 1;

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

```sql

```


