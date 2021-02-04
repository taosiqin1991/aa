

550 游戏玩法分析 系列
activity_table
player_id | device_id | event_date | games_played


1) 所有玩家首次登录的时间及ID
2) 首次登陆之后第二天也登录的玩家数量
3) 玩家总数
4) 在首次登录的第二天再次登录的玩家的比率
5) 描述每一个玩家首次登陆的设备名称
6) 获取每位玩家 第一次登陆平台的日期
7) 同时报告每组玩家和日期，以及玩家到目前为止玩了多少游戏
8)
10)
11)




首次登陆的第二天再次登陆的玩家数 / 总人数
Activity表 内连接 (玩家，首次登陆日期)表
连接条件： ID相同 且 登陆日期 - 首次登陆日期 = 1


首次登录的时间及ID
```sql
-- min + group by
select player_id, min(event_data) first_login
from activity group by player_id;

-- n time login
select player_id, to_char(event_date) first_login
from (
    select a.* ,dense_rank() over(partition by player_id order by event_date) x
    from activity a) tmp
where tmp.x=N;

```


首次登陆之后第二天也登录的玩家数量

先过滤出每个用户的首次登陆日期，然后左关联，筛选次日存在的记录的比例
```sql
/* 364 ms, defeat 99% */
select round(avg(a.event_date is not null), 2) fraction
from(
    select player_id, min(event_date) login
    from activity group by player_id) p
left join activity a
on p.player_id=a.player_id and datediff(a.event_date, p.login)=1;

```


```sql
-- first_login id and time
select player_id, min(event_date) event_date from activity group by player_id;

-- continue second_login player num
select count(*) replay_num from activity a
join(
    select player_id, min(event_date) event_date from activity group by player_id
) b
on a.player_id=b.player_id and datediff(a.event_date, b.event_date)=1;

-- player_sum
select count(distinct player_id) total_num from activity;


-- continue seecond_login player ratio
-- 380 ms defeat 94% 
select round(replay_num/ total_num, 2) fraction
from(
    select count(*) replay_num from activity a
    join(
        select player_id, min(event_date) event_date from activity group by player_id
    ) b
    on a.player_id= b.player_id and datediff(a.event_date, b.event_date)=1   
) re 
cross join(
    select count(distinct player_id) total_num from activity
) tot;
```


获取每位玩家 第一次登陆平台的日期

```sql
#4. 笛卡尔拼接


```

描述每一个玩家首次登陆的设备名称

```sql
/* in, 384 ms*/
select player_id, device_id from activity
where (player_id, event_date) in(select player_id, min(event_date)
                                from activity group by 1);

/* rank 374 ms, 99% */
-- 389 ms
select player_id, device_id
from(
    select player_id, device_id, rank() over(partition by player_id order by event_date) `rank`
    from activity) a
where `rank`=1;

```

同时报告每组玩家和日期，以及玩家到目前为止玩了多少游戏

优先选择窗口函数
```sql
/* 538 ms, defeat 97%， window func */
select player_id, event_date, sum(games_played) over(partition by player_id order by event_date) games_played_so_far from activity;

/*  642 ms, defeat 26%, join */
select a1.player_id, a1.event_date, sum(a2.games_played) games_played_so_far
from activity a1, activity a2
where a1.player_id=a2.player_id and a1.event_date>= a2.event_date
group by 1,2;
```


1107 每日新用户统计

traffic_table:
user_id, activity, activity_date

expect: login_date, user_count

```sql
-- 347 ms 95%
select login_date, count(user_id) user_count
from(
    select user_id, min(activity_date) login_date from traffic
    where activity='login' group by user_id 
    having datediff('2019-06-30', login_date)<=90 ) t
group by login_date;

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


```sql

```

