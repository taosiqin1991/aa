


570 至少有5名直接下属的经理


我们观察Employee表的组成，会发现只要根据ManagerId分组，然后having过滤出5个及以上的非空ManagerId，就可以找到符合条件的经理的Id；
后面通过Id,获取名字就有各种不同的方法了，比如in+子查询，或者自连接join等，输出Name即可

208 ms
```sql
-- find managerid
select managerid from employee e group by managerid having managerid!='null' and count(managerid)>=5;

-- find name of managerid
select name from employee 
where id in(
    select managerid from employee e group by managerid having managerid!='null' and count(managerid)>=5
)
```
可以根据id = managerid，来对Employee表进行自连接，这样可以自然过滤掉managerid为null的数据，然后我们只需根据Id对产生的笛卡尔积进行分组，然后having出现5次及以上的Name，打印即可


inner join
```sql
-- 208 ms
select distinct e1.name from employee e1
inner join employee e2 on e1.id=e2.managerid
group by e1.id having count(e2.managerid) >=5

-- 208 ms
select distinct e1.name from employee e1
inner join employee e2 on e1.id=e2.managerid
group by e2.managerid having count(e2.managerid) >=5

```

176 第二高的薪水
employee_table:
id, salary


考虑异常值，本表可能只有一项记录。ifnull + limit
```sql
-- 144 ms 97%
select ifnull(
    (select distinct salary from employee
    order by salary desc 
    limit 1 offset 1
), null) secondhighestsalary;

```

1468 计算税后工资
salaries_table:
company_id, employee_id, employee_name, salary

每个公司的税率计算依照以下规则: 
如果这个公司员工最高工资不到 1000 ，税率为 0%
如果这个公司员工最高工资在 1000 到 10000 之间，税率为 24%
如果这个公司员工最高工资大于 10000 ，税率为 49%
(此题规则不合理，先忽略)

{"headers":{"Salaries":["company_id","employee_id","employee_name","salary"]},"rows":{"Salaries":[[1,1,"Tony",2000],[1,2,"Pronub",21300],[1,3,"Tyrrox",10800],[2,1,"Pam",300],[2,7,"Bassem",450],[2,9,"Hermione",700],[3,7,"Bocaben",100],[3,2,"Ognjen",2200],[3,13,"Nyancat",3300],[3,15,"Morninngcat",7777]]}}



{"headers":{"Salaries":["company_id","employee_id","employee_name","salary"]},"rows":{"Salaries":[[3,7,"Bocaben",100]]}}



需要连接。连接省不了。
先select出每个公司的tax
做一次join，在Salaries表后加上每个人对于的tax
计算扣除tax后的工资 round(salary * (1-tax),0)

```sql
-- 427 ms 95%
select s.company_id, s.employee_id, s.employee_name, round(s.salary * (1-t.tax),0) as salary
from salaries as s left join (
    select company_id, (
        case
        when max(salary) < 1000 then 0
        when max(salary) between 1000 and 10000 then 0.24
        when max(salary) > 10000 then 0.49
        end
    ) tax from salaries group by company_id
) t 
on s.company_id = t.company_id;

```

1204 最后一个能进入电梯的人
queue_table:
person_id, person_name, weight, turn


```sql
-- 501 ms 98%
select person_name from (
    select person_name, sum(weight) over(order by turn) total_weight from queue
) a 
where total_weight<=1000
order by total_weight desc
limit 1


-- bad 1365 ms 34%
select a.person_name from queue a, queue b
where a.turn >= b.turn
group by a.person_id having sum(b.weight)<=1000
order by a.turn desc
limit 1

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
