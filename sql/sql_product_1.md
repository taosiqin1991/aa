



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


1384 按年度列出销售总额
product_table:
product_id, product_name

sales_table:
product_id, period_start, period_end, average_daily_sales

查询每个产品每年的总销售额，并包含 product_id, product_name 以及 report_year 等信息.


```sql
/*
select case
when year(period_end)=year(period_start)
    then datediff(period_end, period_start)+1
when year(period_end)-year(period_start)=1
    then year(period_start)
end
from
*/

-- 270 ms
with rp_y as(
    select '2018-01-01' year_start, '2018-12-31' year_end, '2018' year
    union
    select '2019-01-01','2019-12-31','2019'
    union
    select '2020-01-01','2020-12-31','2020'
)

select a.product_id, p.product_name, year report_year,
days* average_daily_sales total_amount from(
    select case
    when year(s.period_end)=year(s.period_start) and year(s.period_start)=r.year
        then datediff(s.period_end, s.period_start)+1
    
    when year(s.period_end)-year(s.period_start)=1 and year(s.period_start)=r.year
        then datediff(r.year_end, s.period_start)+1
    
    when year(s.period_end)-year(s.period_start)=1 and year(s.period_end)=r.year
        then datediff(s.period_end, r.year_start)+1

    --- 
    when year(s.period_end)-year(s.period_start)=2 and year(s.period_start)=r.year
        then datediff(r.year_end, s.period_start)+1

    when year(s.period_end)-year(s.period_start)=2 and year(s.period_end)=r.year
        then datediff(s.period_end, r.year_start)+1
    
    when year(s.period_end)-year(s.period_start)=2 and year(s.period_end)=r.year+1
        then 365
    
    end days,
    s.*, r.* from rp_y r
    cross join sales s
) a
join product p on a.product_id=p.product_id
where days is not null
order by product_id, report_year
```

本题核心思路是计算日期范围内占各年份的数量
临时表dates中是遍历生成18年到20年 3年的数据
然后关联count可以计算

select 1+1 as 'he' from dual

下面代码慢，且只适用oracle。mysql报错。
```sql
-- oracle
-- 1236 ms
-- 1889 ms
with dates as(
    select to_date('2018-01-01','yyyy-mm-dd') +rownum-1 as dates from dual
    connect by rownum<=1096),
total as(
    select s.product_id, to_char(de.dates,'yyyy') as report_year, 
    count(1)*max(s.average_daily_sales) as total_amount
    from sales s inner join dates de on de.dates between s.period_start and s.period_end
    group by s.product_id, to_char(de.dates, 'yyyy')
)

select to_char(p.product_id) product_id, p.product_name, t.report_year, t.total_amount
from product p left join total t on p.product_id=t.product_id
order by to_char(p.product_id), t.report_year

```



作者：overme
链接：https://leetcode-cn.com/problems/total-sales-amount-by-year/solution/ben-ti-si-lu-shi-ji-suan-ri-qi-fan-wei-nei-zhan-ge/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



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


1607  没有卖出的卖家
customer_table:
customer_id, customer_name

orders_table:
order_id, sale_date, order_cost, customer_id, seller_id

seller_table:
seller_id, seller_name

查询所有在2020年度没有任何卖出的卖家的名字
year
```sql
-- 533 ms 95%
select seller_name from seller s
left join orders o on s.seller_id=o.seller_id and year(o.sale_date)=2020
where order_id is null
order by seller_name;


-- 533 ms
select seller_name from seller s
left join orders o on s.seller_id=o.seller_id and sale_date between '2020-01-01' and '2020-12-31'
group by s.seller_id
having count(order_id)=0
order by seller_name asc;
```

1364 顾客的可信联系人数量
customer_table:
customer_id, customer_name, email

contacts_table:
user_id, contact_name, contact_email

invoices_table:
invoice_id, price, user_id


按每张invoice查找 顾客名字，金额，顾客联系人数量，可信联系人数量
可信联系人的数量：既是该顾客的联系人又是商店顾客的联系人数量。

```sql
-- 840 ms 93%
-- 880 ms
select invoice_id, c1.customer_name, price,
       count(c2.contact_email) contacts_cnt,
       count(c3.email) trusted_contacts_cnt
from invoices i 
left join customers c1 on i.user_id=c1.customer_id
left join contacts c2 on c1.customer_id=c2.user_id
left join customers c3 on c2.contact_email =c3.email
group by i.invoice_id
order by i.invoice_id;

-- 890 ms
-- 870 ms
select invoice_id, c1.customer_name, price,
       count(c2.contact_email) contacts_cnt,
       count(c3.email) trusted_contacts_cnt
from invoices i 
left join customers c1 on i.user_id=c1.customer_id
left join contacts c2 on c1.customer_id=c2.user_id
left join customers c3 on c2.contact_email =c3.email
group by 1,2,3
order by 1;


```

1549 每件商品的最新订单

customers_table:
customer_id, name

orders_table:
order_id, order_date, customer_id, product_id

products_table:
product_id, product_name, price

expect:
产品名，产品id，订单id，订单日期

Column 'product_id' in IN/ALL/ANY subquery is ambiguous


```sql
-- 1332 ms 93%
select p.product_name, p.product_id, t.order_id, t.order_date
from orders t left join products p
using(product_id)
where (product_id, order_date) in(
    select product_id, max(order_date) order_date from orders o
    group by product_id
)
order by product_name, product_id, order_id;


-- bug on语法错误
select p.product_name, p.product_id, t.order_id, t.order_date
from orders t left join products p
on t.product_id = p.product_id
where (product_id, order_date) in(
    select product_id, max(order_date) order_date from orders o
    group by product_id
)
order by product_name, product_id, order_id;


-- bug 这样写得到的 order_id 是错误的
select p.product_name, p.product_id, t.order_id, t.order_date
from products p inner join (
    select product_id, order_id, max(order_date) order_date from orders o
    group by product_id
) t
on p.product_id=t.product_id
order by p.product_name;


```


1532 最近三笔订单
找到每个用户的最近三笔订单
如果用户的订单少于 3 笔，则返回他的全部订单。
customer_name 升序排列，customer_id 升序排列，order_date 降序排列

期望： 用户名，用户id，订单id，订单日期

customers_table:
customer_id, name

orders_table:
order_id, order_date, customer_id, cost

子查询分组排序，取最近三笔交易
```sql
-- 600 ms 97%
select customer_name, customer_id, order_id, order_date
from(
    select c.name customer_name, o.customer_id, o.order_id, o.order_date,
    rank() over(partition by customer_id order by order_date desc) rn
    from orders o join customers c
    where o.customer_id= c.customer_id
) a
where rn<=3
order by customer_name, customer_id, order_date desc

```

1321 餐馆营业额变化增长
7天为一个窗口
计算以 7 天（某日期 + 该日期前的 6 天）为一个时间段的顾客消费平均值

customer_table:
customer_id, name, visited_on, amount

(customer_id, visited_on 访问日期) 是该表的主键
expect:
visited_on, amount, average_amount

```sql

```

mysql5.7，不支持窗口函数，所以选用自连接
where + group by
```sql
-- 282 ms
select visited_on, amount, average_amount
from(
    select visited_on, 
           round(sum(amount) over(order by visited_on rows 6 preceding), 2) amount,
           round(avg(amount) over(order by visited_on rows 6 preceding), 2) average_amount,
           rank() over(order by visited_on) `rank`
    from (select visited_on, sum(amount) amount from customer group by 1) a
) b
where `rank`>=7


-- 270 ms 96%
select a.visited_on, 
    sum(b.amount) amount,
    round(sum(b.amount)/7, 2) average_amount
from( select distinct visited_on from customer) a join customer b
on datediff(a.visited_on, b.visited_on) between 0 and 6
where a.visited_on >=(select min(visited_on) from customer)+6
group by a.visited_on;

-- 300 ms
select a.visited_on, 
    sum(b.amount) amount,
    round(sum(b.amount)/7, 2) average_amount
from( select distinct visited_on from customer) a join customer b
on datediff(a.visited_on, b.visited_on) between 0 and 6
group by 1
having a.visited_on >=(select min(visited_on)+6 from customer)



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


