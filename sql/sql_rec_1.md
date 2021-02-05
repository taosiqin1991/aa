


1264 页面推荐

friendship_table:
user1_id, user2_id

likes_table:
user_id, page_id

写一段 SQL 向user_id = 1 的用户，推荐其朋友们喜欢的页面。不要推荐该用户已经喜欢的页面。
你返回的结果中不应当包含重复项。

union all
```sql
-- 328 ms
select distinct page_id recommended_page from likes
where user_id in(
    select user2_id as user_id from friendship where user1_id=1
    union all 
    select user1_id as user_id from friendship where user2_id=1
) and page_id not in(
    select page_id from likes where  user_id=1
);

```

case when
```sql
-- 325 ms  96%
select distinct page_id recommended_page from likes
where user_id in(
    select (
        case when user1_id=1 then user2_id
        when user2_id=1 then user1_id
        end
    ) user_id
    from friendship where user1_id=1 or user2_id=1
) and page_id not in(
    select page_id from likes where user_id=1
);


```

```sql


```


1148 文章浏览
查询以找出所有浏览过自己文章的作者，结果按照 id 升序排列。

1149 文章浏览
查询来找出在同一天阅读至少两篇文章的人，结果按照 id 升序排序。
views_table:
article_id, author_id, viewer_id, view_date


```sql
-- 306 ms
select distinct author_id id from views where author_id=viewer_id order by author_id;

```

```sql
-- 285 ms 98%
select distinct viewer_id id from views
group by 1, view_date
having count(distinct article_id)>1
order by 1;


-- 318 ms
select distinct viewer_id id from views
group by view_date, viewer_id
having count(distinct article_id)>=2
order by viewer_id;

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


```sql

```


```sql

```


```sql

```


```sql

```



