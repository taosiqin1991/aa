


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
