

615 平均工资，部门与公司比较

table salary
id, employee_id, amount, pay_date

table employee 外键
employee_id, department_id

need
pay_month, department_id, comparison


用 avg(), case when
分三步
计算公司每个月的平均工资
计算每个部门每个月的平均工资
将上述结果比较

```sql
select A.pay_month, department_id,
case
    when department_avg > company_avg then 'higher'
    when department_avg < company_avg then 'lower'
    else 'same'
end as comparison
from(
    select department_id, avg(amount) as department_avg, date_format(pay_date, '%Y-%m') as pay_month
    from salary join employee on salary.employee_id = employee.employee_id
    group by department_id, pay_month
) as A
join(
    select avg(amount) as company_avg, date_format(pay_date, '%Y-%m') as pay_month
    from salary group by date_format(pay_date, '%Y-%m')
) as B
on A.pay_month = B.pay_month
;

```


618 学生地理信息报告

table student
name, continent

target:
America, Asia, Europe

left join 718 ms
case when 652 ms

同样先是根据洲名分组，对同一洲名的学生进行组内排序编号；

然后分别判断这些学生的所属洲：
如果是美洲，则将该学生的名字赋值给美洲，否则置null；
如果是亚洲，则将该学生的名字赋值给亚洲，否则置null；
如果是欧洲，则将该学生的名字赋值给欧洲，否则置null；

到这一步，就已经实现了列转行。下面需要将列转行的结果根据编号整合一下。

```sql
select 
    max(case when continent='America' then name else null end) as America
    ,max(case when continent='Asia' then name else null end) as Asia
    ,max(case when continent='Europe' then name else null end) as Europe
from(
    select name, continent,row_number() over(partition by continent order by name) px from student
) t
group by px
order by px

```
如果少了t, 则报错， Every derived table must have its own alias


1194 锦标赛优胜者

table Players
player_id, group_id,

table Matches
match_id, first_player, second_player, first_score, second_score

target: need winner
group_id, player_id

总分排名

join all
```sql
select group_id, player_id
from(
    select group_id, player_id, sum(sc) as sc
    from(
        -- frist_score of every player
        select Players.group_id, Players.player_id, sum(Matches.first_score) as sc
        from Players join Matches on Players.player_id = Matches.first_player
        group by Players.player_id

        union all
        -- second_score of every player
            select Players.group_id, Players.player_id, sum(Matches.second_score) as sc
        from Players join Matches on Players.player_id = Matches.second_player
        group by Players.player_id
        
    ) s
    group by player_id
    order by sc desc, player_id
) res
group by group_id

```

group by player_id
order by sc desc, player_id 按


569 员工薪水的中位数

table employee
id, company, salary

target:
id, company, salary(median_s)

```sql
select id, company, salary
from(
    select id, company, salary, 
    row_number() over(partition by company order by salary) as ranking,
    count(id) over(partition by company) as cnt
    from employee
) a
where ranking >=cnt/2 and ranking <=cnt/2+1

```


579 查询员工的累计薪水

自定义前N月内累计薪水 题解法
1.查询每个用户的最大月份
2.过滤掉用户的最大月份数，查询用户月份薪水
3.查询最近前N月份（题目为3,N换3）的数据总和

```sql
--
select b.id as id, b.month as month,
(select sum(e2.salary) from employee e2 where e2.id=b.id and e2.month<=b.month 
and e2.month>b.month-3 order by e2.month desc limit 3) as salary
from(
    select e1.id, e1.month, e1.salary from employee e1,
    (select e.id, max(e.month) as month from employee e group by e.id) a
    where e1.id=a.id and e1.month < a.month
    order by e1.id asc, e1.month desc
) b


-- 查询每个用户的最大月份
select e.id, max(e.month) as mon from employee e group by e.id

-- 过滤掉用户的最大月份数，查询用户月份薪水
select e1.id, e1.month, e1.salary from employee e1,
(select e.id, max(e.month) as mon from employee e group by e.id) a
where e1.id=a.id and e1.month< a.mon
order by e1.id asc, e1.mon desc

-- 查询最近前N月份（题目为3,N换3）的数据总和
select sum(e2.salary) from employee e2
where e2.id = b.id and e2.month <= b.month and e2.month > b.month-3
order by e2.month desc



```


571 给定数字的频率查询中位数
number, frequency

need: median

```sql
select round(avg(number), 1) median
from(
    select number,
           sum(frequency) over(order by number) asc_accumu,
           sum(frequency) over(order by number desc) desc_accumu
    from numbers
    ) t1,
    (
    select sum(frequency) total
    from numbers
    ) t2
where asc_accumu>= total/2 and desc_accumu >= total/2

```


1097 游戏玩法分析 V

玩家留存率

table activity
player_id, device_id, event_date, games_played

need: result
install_dt, installs, day1_retention


1.获得每个用户的安装日期
2.左外连接原表，获得安装日期后一天的数据，
按照安装日期分组，统计装机日期那天的总人数以及后一天的总人数(ID要与装机日期那天一致)

```sql
-- 397 ms
--  ID   sum-min(date)  sum-min(date)+1
select install_dt,count(*) as installs, round(count(a.event_date)/count(*),2) as  day1_retention 
from
(
    select player_id,min(event_date) as install_dt
    from activity
    group by player_id
)t 
left join activity a 
     on a.player_id=t.player_id and a.event_date=t.install_dt+1
group by install_dt


-- bug
select a.install_date, count(a.player_id) as `installs`, count(b.player_id) as `return_cnt`
from(
    select player_id, min(event_date) as `install_date`
    from activity 
    group by player_id
) as a
left join activity as b
on (b.event_date = date_add(a.install_date, interval 1 day) and b.player_id = a.player_id)
group by a.install_date

```
1天后的留存率=return_cnt / installs
```sql

-- ok
select a.event_date as `install_dt`, count(a.player_id) as `installs`,
round(count(c.player_id)/count(a.player_id),2) as `day1_retention`
from activity as a
left join activity as b
    on (a.player_id = b.player_id and a.event_date > b.event_date)
left join activity as c
    on (a.player_id = c.player_id and c.event_date=date_add(a.event_date, interval 1 day)) 
where b.event_date is null
group by a.event_date

```




1127 用户购买平台


```sql
-- 432 ms

SELECT a.spend_date, b.platform, SUM(IF(a.platform = b.platform, amount, 0)) 'total_amount',
COUNT(IF(a.platform = b.platform, TRUE, NULL)) 'total_users' FROM 
(SELECT user_id, spend_date, IF(COUNT(DISTINCT platform) = 2, 'both', platform) 'platform',
SUM(amount) 'amount' FROM spending
GROUP BY user_id, spend_date) a,
(SELECT 'desktop' AS 'platform' UNION
SELECT 'mobile' AS 'platform' UNION
SELECT 'both' AS 'both') b
GROUP BY a.spend_date, b.platform


```

1225 报告系统状态的连续日期

用窗口函数和subdate()来找到分组指标diff

```sql
-- 970 ms
select type as period_state, min(date) as start_date, max(date) as end_date
from
(
    select type, date, subdate(date,row_number()over(partition by type order by date)) as diff
    from
    (
        select 'failed' as type, fail_date as date from Failed
        union all
        select 'succeeded' as type, success_date as date from Succeeded
    ) a
)a
where date between '2019-01-01' and '2019-12-31'
group by type,diff
order by start_date


-- 885 ms
SELECT period_state, MIN(date) AS start_date, MAX(date) AS end_date
FROM
    (SELECT date, period_state, IF(@pre_state=period_state, @id, @id:=@id+1) AS id,            @pre_state:=period_state
    FROM

        (SELECT fail_date AS date, 'failed' AS period_state
        FROM Failed

        UNION ALL

        SELECT success_date AS date, 'succeeded' AS period_state
        FROM Succeeded) A, (SELECT @pre_state:=NULL, @id:=0) B
    WHERE date BETWEEN '2019-01-01' AND '2019-12-31'
    ORDER BY date) C
GROUP BY id


```

1159 市场分析


把过滤条件写在连接条件，可以减少一次子查询

```sql
-- 840 ms best
select user_id as seller_id, if(favorite_brand=item_brand, 'yes', 'no') as 2nd_item_fav_brand
from users u
left join(
    select seller_id, item_brand, rank()over(partition by seller_id order by order_date) as rk
    from orders o
    join items i on o.item_id=i.item_id
    order by seller_id, order_date
) t1
on u.user_id = t1.seller_id and t1.rk=2;

```

MySQL 中，使用 @ 来定义一个变量。比如：@a。
MySQL 中，使用 := 来给变量赋值。比如： @a := 123，表示变量 a 的值为 123。
MySQL 中，if(A, B, C) 表示如果 A 成立， 那么执行并返回 B，否则执行并返回 C。


找到每一个用户按日期顺序卖出的第二件商品的品牌
得到用户卖出的第二件商品的品牌后需要和用户最爱的品牌比较

```sql
-- 1130 ms
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




-- bug
select user_id as seller_id, if (r2.item_brand is null || r2.item_brand != favorite_brand, "no", "yes") as 2nd_item_fav_brand
from users
left join (
    select r1.seller_id, items.item_brand from (
        select 
            @rk := if (@seller = a.seller_id, @rk + 1, 1) as rank,
            @seller := a.seller_id as seller_id, 
            a.item_id
        from (
            select seller_id, item_id
            from orders 
            order by seller_id, order_date
        ) a, (select @seller := -1, @rk := 0) b) r1
    join items 
    on r1.item_id = items.item_id
    where r1.rank = 2
) r2 on user_id = r2.seller_id;

```



```sql


```


```sql

```

```sql

```


```sql
select 


```

```sql
select 


```

```sql
select 


```


```sql
select 


```


```sql
select 


```
