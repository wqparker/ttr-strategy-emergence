# USA map data checklist

Generated from `src/ttr/data/usa.json` by `scripts/board_checklist.py`.
Data status: **UNVERIFIED**.

Tick each row once it matches the physical board or cards. Fix mistakes in the
JSON, regenerate this file, and set `"verified": true` once every row is ticked.

## Summary (36 cities, 100 routes, 30 tickets)

| color | routes | train spaces |
|---|---|---|
| black | 7 | 27 |
| blue | 7 | 27 |
| gray | 44 | 93 |
| green | 7 | 27 |
| orange | 7 | 27 |
| purple | 7 | 27 |
| red | 7 | 27 |
| white | 7 | 27 |
| yellow | 7 | 27 |

Total train spaces: 309. Double routes: 22.

## Routes (alphabetical by first city)

| ✓ | from | to | length | color | double |
|---|---|---|---|---|---|
| [ ] | Atlanta | Charleston | 2 | gray |  |
| [ ] | Atlanta | Miami | 5 | blue |  |
| [ ] | Atlanta | Nashville | 1 | gray |  |
| [ ] | Atlanta | New Orleans | 4 | orange | yes |
| [ ] | Atlanta | New Orleans | 4 | yellow | yes |
| [ ] | Atlanta | Raleigh | 2 | gray | yes |
| [ ] | Atlanta | Raleigh | 2 | gray | yes |
| [ ] | Boston | Montreal | 2 | gray | yes |
| [ ] | Boston | Montreal | 2 | gray | yes |
| [ ] | Boston | New York | 2 | red | yes |
| [ ] | Boston | New York | 2 | yellow | yes |
| [ ] | Calgary | Helena | 4 | gray |  |
| [ ] | Calgary | Seattle | 4 | gray |  |
| [ ] | Calgary | Vancouver | 3 | gray |  |
| [ ] | Calgary | Winnipeg | 6 | white |  |
| [ ] | Charleston | Miami | 4 | purple |  |
| [ ] | Charleston | Raleigh | 2 | gray |  |
| [ ] | Chicago | Duluth | 3 | red |  |
| [ ] | Chicago | Omaha | 4 | blue |  |
| [ ] | Chicago | Pittsburgh | 3 | black | yes |
| [ ] | Chicago | Pittsburgh | 3 | orange | yes |
| [ ] | Chicago | Saint Louis | 2 | green | yes |
| [ ] | Chicago | Saint Louis | 2 | white | yes |
| [ ] | Chicago | Toronto | 4 | white |  |
| [ ] | Dallas | El Paso | 4 | red |  |
| [ ] | Dallas | Houston | 1 | gray | yes |
| [ ] | Dallas | Houston | 1 | gray | yes |
| [ ] | Dallas | Little Rock | 2 | gray |  |
| [ ] | Dallas | Oklahoma City | 2 | gray | yes |
| [ ] | Dallas | Oklahoma City | 2 | gray | yes |
| [ ] | Denver | Helena | 4 | green |  |
| [ ] | Denver | Kansas City | 4 | black | yes |
| [ ] | Denver | Kansas City | 4 | orange | yes |
| [ ] | Denver | Oklahoma City | 4 | red |  |
| [ ] | Denver | Omaha | 4 | purple |  |
| [ ] | Denver | Phoenix | 5 | white |  |
| [ ] | Denver | Salt Lake City | 3 | red | yes |
| [ ] | Denver | Salt Lake City | 3 | yellow | yes |
| [ ] | Denver | Santa Fe | 2 | gray |  |
| [ ] | Duluth | Helena | 6 | orange |  |
| [ ] | Duluth | Omaha | 2 | gray | yes |
| [ ] | Duluth | Omaha | 2 | gray | yes |
| [ ] | Duluth | Sault St. Marie | 3 | gray |  |
| [ ] | Duluth | Toronto | 6 | purple |  |
| [ ] | Duluth | Winnipeg | 4 | black |  |
| [ ] | El Paso | Houston | 6 | green |  |
| [ ] | El Paso | Los Angeles | 6 | black |  |
| [ ] | El Paso | Oklahoma City | 5 | yellow |  |
| [ ] | El Paso | Phoenix | 3 | gray |  |
| [ ] | El Paso | Santa Fe | 2 | gray |  |
| [ ] | Helena | Omaha | 5 | red |  |
| [ ] | Helena | Salt Lake City | 3 | purple |  |
| [ ] | Helena | Seattle | 6 | yellow |  |
| [ ] | Helena | Winnipeg | 4 | blue |  |
| [ ] | Houston | New Orleans | 2 | gray |  |
| [ ] | Kansas City | Oklahoma City | 2 | gray | yes |
| [ ] | Kansas City | Oklahoma City | 2 | gray | yes |
| [ ] | Kansas City | Omaha | 1 | gray | yes |
| [ ] | Kansas City | Omaha | 1 | gray | yes |
| [ ] | Kansas City | Saint Louis | 2 | blue | yes |
| [ ] | Kansas City | Saint Louis | 2 | purple | yes |
| [ ] | Las Vegas | Los Angeles | 2 | gray |  |
| [ ] | Las Vegas | Salt Lake City | 3 | orange |  |
| [ ] | Little Rock | Nashville | 3 | white |  |
| [ ] | Little Rock | New Orleans | 3 | green |  |
| [ ] | Little Rock | Oklahoma City | 2 | gray |  |
| [ ] | Little Rock | Saint Louis | 2 | gray |  |
| [ ] | Los Angeles | Phoenix | 3 | gray |  |
| [ ] | Los Angeles | San Francisco | 3 | purple | yes |
| [ ] | Los Angeles | San Francisco | 3 | yellow | yes |
| [ ] | Miami | New Orleans | 6 | red |  |
| [ ] | Montreal | New York | 3 | blue |  |
| [ ] | Montreal | Sault St. Marie | 5 | black |  |
| [ ] | Montreal | Toronto | 3 | gray |  |
| [ ] | Nashville | Pittsburgh | 4 | yellow |  |
| [ ] | Nashville | Raleigh | 3 | black |  |
| [ ] | Nashville | Saint Louis | 2 | gray |  |
| [ ] | New York | Pittsburgh | 2 | green | yes |
| [ ] | New York | Pittsburgh | 2 | white | yes |
| [ ] | New York | Washington | 2 | black | yes |
| [ ] | New York | Washington | 2 | orange | yes |
| [ ] | Oklahoma City | Santa Fe | 3 | blue |  |
| [ ] | Phoenix | Santa Fe | 3 | gray |  |
| [ ] | Pittsburgh | Raleigh | 2 | gray |  |
| [ ] | Pittsburgh | Saint Louis | 5 | green |  |
| [ ] | Pittsburgh | Toronto | 2 | gray |  |
| [ ] | Pittsburgh | Washington | 2 | gray |  |
| [ ] | Portland | Salt Lake City | 6 | blue |  |
| [ ] | Portland | San Francisco | 5 | green | yes |
| [ ] | Portland | San Francisco | 5 | purple | yes |
| [ ] | Portland | Seattle | 1 | gray | yes |
| [ ] | Portland | Seattle | 1 | gray | yes |
| [ ] | Raleigh | Washington | 2 | gray | yes |
| [ ] | Raleigh | Washington | 2 | gray | yes |
| [ ] | Salt Lake City | San Francisco | 5 | orange | yes |
| [ ] | Salt Lake City | San Francisco | 5 | white | yes |
| [ ] | Sault St. Marie | Toronto | 2 | gray |  |
| [ ] | Sault St. Marie | Winnipeg | 6 | gray |  |
| [ ] | Seattle | Vancouver | 1 | gray | yes |
| [ ] | Seattle | Vancouver | 1 | gray | yes |

## Destination tickets

| ✓ | from | to | points |
|---|---|---|---|
| [ ] | Atlanta | Montreal | 9 |
| [ ] | Atlanta | New York | 6 |
| [ ] | Atlanta | San Francisco | 17 |
| [ ] | Boston | Miami | 12 |
| [ ] | Calgary | Phoenix | 13 |
| [ ] | Calgary | Salt Lake City | 7 |
| [ ] | Chicago | Los Angeles | 16 |
| [ ] | Chicago | New Orleans | 7 |
| [ ] | Chicago | Santa Fe | 9 |
| [ ] | Dallas | New York | 11 |
| [ ] | Denver | El Paso | 4 |
| [ ] | Denver | Pittsburgh | 11 |
| [ ] | Duluth | El Paso | 10 |
| [ ] | Duluth | Houston | 8 |
| [ ] | Helena | Los Angeles | 8 |
| [ ] | Houston | Kansas City | 5 |
| [ ] | Houston | Winnipeg | 12 |
| [ ] | Little Rock | Winnipeg | 11 |
| [ ] | Los Angeles | Miami | 20 |
| [ ] | Los Angeles | New York | 21 |
| [ ] | Los Angeles | Seattle | 9 |
| [ ] | Miami | Toronto | 10 |
| [ ] | Montreal | New Orleans | 13 |
| [ ] | Montreal | Vancouver | 20 |
| [ ] | Nashville | Portland | 17 |
| [ ] | Nashville | Sault St. Marie | 8 |
| [ ] | New York | Seattle | 22 |
| [ ] | Oklahoma City | Sault St. Marie | 9 |
| [ ] | Phoenix | Portland | 11 |
| [ ] | Santa Fe | Vancouver | 13 |
